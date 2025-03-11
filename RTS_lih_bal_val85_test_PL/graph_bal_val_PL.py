"""
Для сохранения графиков в файлы. Лиховидов. Бинарка.
С балансировкой классов добавлением рандомных, где нет совпадения по фичам с противоположным классом.
Лучшая модель сохраняется по Profit - Loss критерию.
Только валидация.
"""
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
import matplotlib.pyplot as plt


db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_day_full.db')
model_path = Path(r"RTS_lih_bal_val85_test_PL\best_model_profit_loss_val.pth")

for counter in range(1, 101):
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(counter)  # Устанавливаем одинаковый seed

    with sqlite3.connect(db_path) as conn:
        df_fut = pd.read_sql_query(
            """
            SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME 
            FROM Day 
            WHERE TRADEDATE BETWEEN '2014-01-01' AND '2024-01-01' 
            ORDER BY TRADEDATE
            """,
            conn
        )

    def encode_candle(row):
        open_, low, high, close = row['OPEN'], row['LOW'], row['HIGH'], row['CLOSE']
        direction = 1 if close > open_ else (0 if close < open_ else 2)
        upper_shadow = high - max(open_, close)
        lower_shadow = min(open_, close) - low
        body = abs(close - open_)

        def classify_shadow(shadow, body):
            return 0 if shadow < 0.1 * body else (1 if shadow < 0.5 * body else 2)

        return (f"{direction}{classify_shadow(upper_shadow, body)}"
                f"{classify_shadow(lower_shadow, body)}")


    df_fut['CANDLE_CODE'] = df_fut.apply(encode_candle, axis=1)

    unique_codes = sorted(df_fut['CANDLE_CODE'].unique())
    code_to_int = {code: i for i, code in enumerate(unique_codes)}
    df_fut['CANDLE_INT'] = df_fut['CANDLE_CODE'].map(code_to_int)

    window_size = 20
    predict_offset = 1

    X, y = [], []
    for i in range(len(df_fut) - window_size - predict_offset):
        X.append(df_fut['CANDLE_INT'].iloc[i:i + window_size].values)
        y.append(
            1 if df_fut['CLOSE'].iloc[i + window_size + predict_offset] >
                 df_fut['OPEN'].iloc[i + window_size + predict_offset] else 0
        )

    X, y = np.array(X), np.array(y)

    split = int(0.85 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # === Балансировка классов === ----------------------------------------------------------------
    # Объединяем X_train и y_train в один DataFrame
    df_train = pd.DataFrame(X_train)
    df_train['TARGET'] = y_train  # Добавляем колонку с целевой меткой

    # Проверяем начальное распределение классов
    class_counts = df_train['TARGET'].value_counts()
    print("Распределение перед балансировкой:\n", class_counts)

    # Определяем миноритарный и мажоритарный классы
    min_class = class_counts.idxmin()
    max_class = class_counts.idxmax()

    # Разделяем по классам
    df_minority = df_train[df_train['TARGET'] == min_class]
    df_majority = df_train[df_train['TARGET'] == max_class]

    # Убираем из миноритарного класса строки, которые полностью совпадают с мажоритарным классом по фичам
    df_minority_unique = df_minority.loc[
        ~df_minority.drop(columns=['TARGET']).apply(tuple, axis=1).isin(
            df_majority.drop(columns=['TARGET']).apply(tuple, axis=1)
        )]

    # Дублируем ОТФИЛЬТРОВАННЫЕ примеры редкого класса
    df_minority_upsampled = resample(df_minority_unique,
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=42)

    # Объединяем оба класса и перемешиваем
    df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42)

    # Разделяем обратно на X_train и y_train
    X_train = df_balanced.drop(columns=['TARGET']).values
    y_train = df_balanced['TARGET'].values

    # Проверяем новое распределение
    print("Распределение после балансировки:\n", pd.Series(y_train).value_counts())
    # Конец балансировки --------------------------------------------------------------------------

    class CandlestickDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.long)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


    def seed_worker(worker_id):
        np.random.seed(42 + worker_id)
        random.seed(42 + worker_id)


    train_dataset = CandlestickDataset(X_train, y_train)
    test_dataset = CandlestickDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, worker_init_fn=seed_worker)


    # === 5. СОЗДАНИЕ НЕЙРОСЕТИ (LSTM) ===
    class CandleLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
            super(CandleLSTM, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.lstm(x)
            x = self.fc(x[:, -1, :])
            return self.sigmoid(x)


    # === 6. ОБУЧЕНИЕ МОДЕЛИ С ОПТИМИЗАЦИЕЙ NET PIPS ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CandleLSTM(vocab_size=len(unique_codes), embedding_dim=8, hidden_dim=32,
                       output_dim=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_net_pips = float('-inf')  # Храним лучший критерий profit - loss
    epoch_best = 0
    # model_path = Path("best_model_profit_loss.pth")
    early_stop_epochs = 200
    epochs_no_improve = 0

    epochs = 2000
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # === Оценка модели по критерию Profit - Loss на тестовой выборке после каждой эпохи ===
        model.eval()
        total_profit = 0
        total_loss_pips = 0

        with torch.no_grad():
            batch_start = split  # Начало тестовой выборки
            for batch_idx, (X_batch, _) in enumerate(test_loader):
                X_batch = X_batch.to(device)
                y_pred = model(X_batch).squeeze().round()

                # Рассчитываем индексы для текущего батча
                batch_indices = range(batch_start + batch_idx * len(y_pred),
                                      batch_start + (batch_idx + 1) * len(y_pred))

                for i, idx in enumerate(batch_indices):
                    if idx + window_size + predict_offset < len(df_fut):
                        open_price = df_fut.iloc[idx + window_size]['OPEN']
                        close_price = df_fut.iloc[idx + window_size + predict_offset]['CLOSE']

                        if y_pred[i] == 1:  # Прогноз роста
                            if close_price > open_price:
                                total_profit += close_price - open_price  # Профит
                            else:
                                total_loss_pips += open_price - close_price  # Убыток
                        else:  # Прогноз падения
                            if close_price < open_price:
                                total_profit += open_price - close_price  # Профит
                            else:
                                total_loss_pips += close_price - open_price  # Убыток

        net_pips = total_profit - total_loss_pips

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {total_loss / len(train_loader):.4f}, "
            f"Net Pips: {int(net_pips)}, "
            f"Best net pips: {best_net_pips}, "
            f"Epoch best pips: {epoch_best}, "
            f"seed: {counter}"
        )

        # === Сохранение лучшей модели по net_pips ===
        if net_pips > best_net_pips:
            best_net_pips = net_pips
            epochs_no_improve = 0
            epoch_best = epoch + 1
            torch.save(model.state_dict(), model_path)
            print(f"✅ Model saved with Net Pips: {int(best_net_pips)}")
        else:
            epochs_no_improve += 1

        # === Ранняя остановка ===
        if epochs_no_improve >= early_stop_epochs:
            print(f"🛑 Ранняя остановка на эпохе {epoch + 1}")
            break

    # === 7. ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ И ТЕСТ ===
    print("\n🔹 Loading best model for final evaluation...")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch).squeeze().round()
            correct += (y_pred == y_batch).sum().item()
            total += y_batch.size(0)

    final_accuracy = correct / total
    print(f"🏆 Final Test Accuracy: {final_accuracy:.2%}")

    # --------------------------------------------------------------------------------------------
    # === 1. ЗАГРУЗКА ДАННЫХ ===

    with sqlite3.connect(db_path) as conn:
        df_fut = pd.read_sql_query(
            """
            SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME 
            FROM Day 
            WHERE TRADEDATE BETWEEN '2014-01-01' AND '2024-01-01' 
            ORDER BY TRADEDATE
            """,
            conn
        )


    # === 2. ФУНКЦИЯ КОДИРОВАНИЯ СВЕЧЕЙ (ЛИХОВИДОВ) ===
    def encode_candle(row):
        open_, low, high, close = row['OPEN'], row['LOW'], row['HIGH'], row['CLOSE']

        direction = 1 if close > open_ else (0 if close < open_ else 2)
        upper_shadow = high - max(open_, close)
        lower_shadow = min(open_, close) - low
        body = abs(close - open_)

        def classify_shadow(shadow, body):
            return 0 if shadow < 0.1 * body else (1 if shadow < 0.5 * body else 2)

        return f"{direction}{classify_shadow(upper_shadow, body)}{classify_shadow(lower_shadow, body)}"


    df_fut['CANDLE_CODE'] = df_fut.apply(encode_candle, axis=1)

    # === 3. ПРЕОБРАЗОВАНИЕ КОДОВ В ЧИСЛА ===
    unique_codes = sorted(df_fut['CANDLE_CODE'].unique())
    code_to_int = {code: i for i, code in enumerate(unique_codes)}
    df_fut['CANDLE_INT'] = df_fut['CANDLE_CODE'].map(code_to_int)

    window_size = 20

    # # === 4. ОПРЕДЕЛЕНИЕ МОДЕЛИ (ДОЛЖНА СОВПАДАТЬ С ОБУЧЕННОЙ) ===
    # class CandleLSTM(nn.Module):
    #     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
    #         super(CandleLSTM, self).__init__()
    #         self.embedding = nn.Embedding(vocab_size, embedding_dim)
    #         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    #         self.fc = nn.Linear(hidden_dim, output_dim)
    #         self.sigmoid = nn.Sigmoid()
    #
    #     def forward(self, x):
    #         x = self.embedding(x)
    #         x, _ = self.lstm(x)
    #         x = self.fc(x[:, -1, :])
    #         return self.sigmoid(x)

    # === 5. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model_path = Path(r"best_model_graph_RTS_bal_01.pth")  # Уже есть значение
    model = CandleLSTM(vocab_size=len(unique_codes), embedding_dim=8, hidden_dim=32,
                       output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 6. ПРОГНОЗИРОВАНИЕ ===
    predictions = []
    with torch.no_grad():
        for i in range(len(df_fut) - window_size):
            sequence = torch.tensor(
                df_fut['CANDLE_INT'].iloc[i:i + window_size].values, dtype=torch.long
            ).unsqueeze(0).to(device)
            pred = model(sequence).item()
            predictions.append(1 if pred > 0.5 else 0)

    # Заполняем колонку PREDICTION (первые window_size значений - NaN)
    df_fut['PREDICTION'] = [None] * window_size + predictions

    # # === 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
    # predictions_file = Path(r"predictions_graph_RTS_01.csv")
    # df_fut.to_csv(predictions_file, index=False)
    # print(f"✅ Прогнозы сохранены в '{predictions_file}'")

    # -------------------------------------------------------------------------------------
    # # === 1. ЗАГРУЗКА ФАЙЛА И ОТБОР ПОСЛЕДНИХ 20% ===
    # df = pd.read_csv(predictions_file)

    split = int(len(df_fut) * 0.85)  # 80% - обучающая выборка, 20% - тестовая
    df = df_fut.iloc[split:].copy()  # Берем последние 20%

    # === 3. РАСЧЁТ РЕЗУЛЬТАТОВ ПРОГНОЗА ===
    def calculate_result(row):
        if pd.isna(row["PREDICTION"]):  # Если NaN после сдвига
            return 0  # Можно удалить или оставить 0

        true_direction = 1 if row["CLOSE"] > row["OPEN"] else 0
        predicted_direction = row["PREDICTION"]

        difference = abs(row["CLOSE"] - row["OPEN"])
        return difference if true_direction == predicted_direction else -difference


    df["RESULT"] = df.apply(calculate_result, axis=1)

    # === 4. ПОСТРОЕНИЕ КУМУЛЯТИВНОГО ГРАФИКА ===
    df["CUMULATIVE_RESULT"] = df["RESULT"].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(df["TRADEDATE"], df["CUMULATIVE_RESULT"], label="Cumulative Result",
             color="b")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Result")
    plt.title(f"Cumulative Sum RTS. set_seed={counter}, "
              f"Best pips: {int(best_net_pips)}, "
              f"Epoch best pips: {epoch_best}, "
              f"Final Test Accuracy: {final_accuracy:.2%}")
    plt.legend()
    plt.grid()

    plt.xticks(df["TRADEDATE"][::10], rotation=90)
    # Сохранение графика в файл
    img_path = Path(fr"RTS_lih_bal_val85_test_PL\img_RTS_net_pips_val\s_{counter}_RTS.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"✅ График сохранен в файл: '{img_path}' \n")
    # plt.show()
