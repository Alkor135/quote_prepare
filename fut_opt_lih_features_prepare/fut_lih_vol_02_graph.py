"""
Для сохранения графиков в файлы. Лиховидов и объемы. Бинарка.
Квантиль объема теперь рассчитывается на основе 10 предыдущих свечей, а не на всей выборке
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
import matplotlib.pyplot as plt

for counter in range(1, 99):
    # === 1. ФИКСАЦИЯ СЛУЧАЙНЫХ ЧИСЕЛ ДЛЯ ДЕТЕРМИНИРОВАННОСТИ ===
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    set_seed(counter)  # Устанавливаем одинаковый seed

    # === 1. ЗАГРУЗКА ДАННЫХ ===
    db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day.db')

    with sqlite3.connect(db_path) as conn:
        df_fut = pd.read_sql_query(
            "SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME FROM Futures",
            conn
        )

    # === 2. ФУНКЦИЯ КОДИРОВАНИЯ СВЕЧЕЙ (ЛИХОВИДОВ) С ДИНАМИЧЕСКИМ КОДИРОВАНИЕМ ОБЪЕМА ===
    window_size_volume = 10  # Количество свечей для расчета квантилей

    def encode_volume(index, volume):
        if index < window_size_volume:
            return '1'  # Для первых 10 свечей устанавливаем средний объем

        past_volumes = df_fut['VOLUME'].iloc[index - window_size_volume:index]
        low_quantile = past_volumes.quantile(0.33)
        high_quantile = past_volumes.quantile(0.66)

        if volume <= low_quantile:
            return '0'  # Низкий объем
        elif volume <= high_quantile:
            return '1'  # Средний объем
        else:
            return '2'  # Высокий объем


    def encode_candle(index, row):
        open_, low, high, close, volume = row['OPEN'], row['LOW'], row['HIGH'], row['CLOSE'], row['VOLUME']

        if close > open_:
            direction = 1
        elif close < open_:
            direction = 0
        else:
            direction = 2

        upper_shadow = high - max(open_, close)
        lower_shadow = min(open_, close) - low
        body = abs(close - open_)

        def classify_shadow(shadow, body):
            if shadow < 0.1 * body:
                return 0
            elif shadow < 0.5 * body:
                return 1
            else:
                return 2

        upper_code = classify_shadow(upper_shadow, body)
        lower_code = classify_shadow(lower_shadow, body)

        volume_code = encode_volume(index, volume)  # Код объема

        return f"{direction}{upper_code}{lower_code}{volume_code}"  # Код свечи + объем


    df_fut['CANDLE_CODE'] = [encode_candle(i, row) for i, row in df_fut.iterrows()]

    # === 3. ПОДГОТОВКА ДАННЫХ ===
    unique_codes = sorted(df_fut['CANDLE_CODE'].unique())
    code_to_int = {code: i for i, code in enumerate(unique_codes)}
    df_fut['CANDLE_INT'] = df_fut['CANDLE_CODE'].map(code_to_int)

    window_size = 20
    predict_offset = 1

    X, y = [], []
    for i in range(len(df_fut) - window_size - predict_offset):
        # Add features
        X.append(df_fut['CANDLE_INT'].iloc[i:i + window_size].values)

        # Новый таргет: 1 — рост, 0 — падение
        y.append(
            1 if df_fut['CLOSE'].iloc[i + window_size + predict_offset] >
                 df_fut['CLOSE'].iloc[i + window_size] else 0
        )

    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]


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

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, worker_init_fn=seed_worker
    )


    # === 4. СОЗДАНИЕ НЕЙРОСЕТИ (LSTM) ===
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


    # === 5. ОБУЧЕНИЕ МОДЕЛИ С СОХРАНЕНИЕМ ЛУЧШЕЙ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CandleLSTM(
        vocab_size=len(unique_codes), embedding_dim=8, hidden_dim=32, output_dim=1
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0
    epoch_best_accuracy = 0
    model_path = "bm_fut_lih_vol_02.pth"
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

        # === Проверка на тесте после каждой эпохи ===
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch).squeeze().round()
                correct += (y_pred == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = correct / total
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {total_loss / len(train_loader):.4f}, "
            f"Test Accuracy: {accuracy:.2%}, "
            f"Best accuracy: {best_accuracy:.2%}, "
            f"Epoch best accuracy: {epoch_best_accuracy}, "
            f"seed: {counter}"
        )

        # === Сохранение лучшей модели ===
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_no_improve = 0
            epoch_best_accuracy = epoch + 1
            torch.save(model.state_dict(), model_path)
            print(f"✅ Model saved with accuracy: {best_accuracy:.2%}")
        else:
            epochs_no_improve += 1

        # === Ранняя остановка ===
        if epochs_no_improve >= early_stop_epochs:
            print(f"🛑 Early stopping at epoch {epoch + 1}")
            break

    # === 6. ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ И ТЕСТ ===
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
    db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day.db')

    with sqlite3.connect(db_path) as conn:
        df_fut = pd.read_sql_query(
            "SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME FROM Futures",
            conn
        )

    # === 2. ФУНКЦИЯ КОДИРОВАНИЯ СВЕЧЕЙ (ЛИХОВИДОВ) С ДИНАМИЧЕСКИМ КОДИРОВАНИЕМ ОБЪЕМА ===
    window_size_volume = 10  # Количество свечей для расчета квантилей


    def encode_volume(index, volume):
        if index < window_size_volume:
            return '1'  # Для первых 10 свечей устанавливаем средний объем

        past_volumes = df_fut['VOLUME'].iloc[index - window_size_volume:index]
        low_quantile = past_volumes.quantile(0.33)
        high_quantile = past_volumes.quantile(0.66)

        if volume <= low_quantile:
            return '0'  # Низкий объем
        elif volume <= high_quantile:
            return '1'  # Средний объем
        else:
            return '2'  # Высокий объем


    def encode_candle(index, row):
        open_, low, high, close, volume = (
            row['OPEN'], row['LOW'], row['HIGH'], row['CLOSE'], row['VOLUME']
        )

        if close > open_:
            direction = 1
        elif close < open_:
            direction = 0
        else:
            direction = 2

        upper_shadow = high - max(open_, close)
        lower_shadow = min(open_, close) - low
        body = abs(close - open_)

        def classify_shadow(shadow, body):
            if shadow < 0.1 * body:
                return 0
            elif shadow < 0.5 * body:
                return 1
            else:
                return 2

        upper_code = classify_shadow(upper_shadow, body)
        lower_code = classify_shadow(lower_shadow, body)

        volume_code = encode_volume(index, volume)  # Код объема

        return f"{direction}{upper_code}{lower_code}{volume_code}"  # Код свечи + объем


    df_fut['CANDLE_CODE'] = [encode_candle(i, row) for i, row in df_fut.iterrows()]

    # === 3. ПРЕОБРАЗОВАНИЕ КОДОВ В ЧИСЛА ===
    unique_codes = sorted(df_fut['CANDLE_CODE'].unique())
    code_to_int = {code: i for i, code in enumerate(unique_codes)}
    df_fut['CANDLE_INT'] = df_fut['CANDLE_CODE'].map(code_to_int)

    window_size = 20


    # === 4. ОПРЕДЕЛЕНИЕ МОДЕЛИ (ДОЛЖНА СОВПАДАТЬ С ОБУЧЕННОЙ) ===
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


    # === 5. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "bm_fut_lih_vol_02.pth"
    model = CandleLSTM(
        vocab_size=len(unique_codes), embedding_dim=8, hidden_dim=32, output_dim=1
    ).to(device)
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

    # === 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
    df_fut.to_csv("predictions_fut_lih_vol_02.csv", index=False)
    print("✅ Прогнозы сохранены в 'predictions_fut_lih_vol_02'")

    # -------------------------------------------------------------------------------------
    # === 1. ЗАГРУЗКА ФАЙЛА И ОТБОР ПОСЛЕДНИХ 20% ===
    df = pd.read_csv("predictions_fut_lih_vol_02.csv")

    split = int(len(df) * 0.8)  # 80% - обучающая выборка, 20% - тестовая
    df = df.iloc[split:].copy()  # Берем последние 20%

    # === 2. СМЕЩЕНИЕ ПРОГНОЗА НА ОДИН БАР ВПЕРЁД ===
    df["PREDICTION_SHIFTED"] = df["PREDICTION"].shift(1)  # Смещаем вверх


    # df

    # === 3. РАСЧЁТ РЕЗУЛЬТАТОВ ПРОГНОЗА ===
    def calculate_result(row):
        if pd.isna(row["PREDICTION_SHIFTED"]):  # Если NaN после сдвига
            return 0  # Можно удалить или оставить 0

        true_direction = 1 if row["CLOSE"] > row["OPEN"] else 0
        predicted_direction = row["PREDICTION_SHIFTED"]

        difference = abs(row["CLOSE"] - row["OPEN"])
        return difference if true_direction == predicted_direction else -difference


    df["RESULT"] = df.apply(calculate_result, axis=1)

    # === 4. ПОСТРОЕНИЕ КУМУЛЯТИВНОГО ГРАФИКА ===
    df["CUMULATIVE_RESULT"] = df["RESULT"].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(df["TRADEDATE"], df["CUMULATIVE_RESULT"], label="Cumulative Result", color="b")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Result")
    plt.title(f"Cumulative Sum of Prediction Accuracy (fut_lih_vol_02) set_seed={counter}, "
              f"Best accuracy: {best_accuracy:.2%}, "
              f"Epoch best accuracy: {epoch_best_accuracy}")
    plt.legend()
    plt.grid()

    # plt.xticks(rotation=45)
    # plt.xticks(df["TRADEDATE"][::10], rotation=45)
    plt.xticks(df["TRADEDATE"][::10], rotation=90)
    # Сохранение графика в файл
    plt.savefig(fr"img_lih_vol_02/fut_lih_vol_02_{counter}.png", dpi=300, bbox_inches='tight')
    print(f"✅ График сохранен в файл: 'img_lih_vol_02/fut_lih_vol_02_{counter}.png' \n")
    # plt.show()
