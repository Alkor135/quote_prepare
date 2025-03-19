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
import json
import os


# === СОЗДАНИЕ НЕЙРОСЕТИ (LSTM) ===
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


# === ФИКСАЦИЯ СЛУЧАЙНЫХ ЧИСЕЛ ДЛЯ ДЕТЕРМИНИРОВАННОСТИ ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# === ФУНКЦИЯ КОДИРОВАНИЯ СВЕЧЕЙ (ЛИХОВИДОВ) ===
def encode_candle(row):
    open_, low, high, close = row['OPEN'], row['LOW'], row['HIGH'], row['CLOSE']

    if close > open_:
        direction = 1  # Бычья свеча
    elif close < open_:
        direction = 0  # Медвежья свеча
    else:
        direction = 2  # Дожи

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

    return f"{direction}{upper_code}{lower_code}"


# === Функция расчета P/L (по предсказанному направлению) ===
def calculate_pnl(y_preds, open_prices, close_prices):
    pnl = 0
    for i in range(len(y_preds)):
        if y_preds[i] > 0.5:  # Покупка (LONG)
            pnl += close_prices[i] - open_prices[i]
        else:  # Продажа (SHORT)
            pnl += open_prices[i] - close_prices[i]
    return pnl  # Итоговая прибыль


# === 1. ОПРЕДЕЛЕНИЯ ===
# Установка рабочей директории в папку, где находится файл скрипта
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Загрузка полного словаря для преобразования кода свечи в числовой формат
with open("code_full_int.json", "r") as f:
    code_to_int = json.load(f)

db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day_2014.db')

for counter in range(101, 201):
    set_seed(counter)  # Устанавливаем одинаковый seed

    # === 2. ЗАГРУЗКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ И ВАЛИДАЦИИ ===
    with sqlite3.connect(db_path) as conn:
        df_fut = pd.read_sql_query(
            """
            SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME 
            FROM Futuures 
            WHERE TRADEDATE BETWEEN '2014-01-01' AND '2024-01-01' 
            ORDER BY TRADEDATE
            """,
            conn
        )

    # Создание кодов свечей по Лиховидову
    df_fut['CANDLE_CODE'] = df_fut.apply(encode_candle, axis=1)

    # === 3. ПОДГОТОВКА ДАННЫХ ===
    # Преобразуем свечные коды в числовой формат (список уникальных кодов)
    df_fut['CANDLE_INT'] = df_fut['CANDLE_CODE'].map(code_to_int)

    # Создание колонки направления.
    df_fut['DIRECTION'] = (df_fut['CLOSE'] > df_fut['OPEN']).astype(int)

    # Создание колонок с признаками
    for i in range(1, 21):
        df_fut[f'CI_{i}'] = df_fut['CANDLE_INT'].shift(i).astype('Int64')

    df_fut = df_fut.dropna().reset_index(drop=True)

    # Создание дата сетов
    feature_columns = [col for col in df_fut.columns if col.startswith('CI')]
    X = df_fut[feature_columns]
    y = df_fut['DIRECTION']
    X, y = np.array(X), np.array(y)

    # Разделение на train/test
    split = int(0.85 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # === 4. Балансировка классов === -------------------------------------------------------------
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

    # Убираем из миноритарного класса строки, которые полностью совпадают 
    # с мажоритарным классом по фичам  
    df_minority_unique = df_minority.loc[  
        ~df_minority.drop(columns=['TARGET']).apply(tuple, axis=1).isin(  
            df_majority.drop(columns=['TARGET']).apply(tuple, axis=1)  
        )]  
    
    # Дублируем ОТФИЛЬТРОВАННЫЕ примеры редкого класса рандомно
    df_minority_upsampled = resample(df_minority_unique,  
                                    replace=True,  
                                    n_samples=len(df_majority),  
                                    random_state=42)  

    # Объединяем оба класса и перемешиваем
    df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42)

    # Разделяем обратно на X_train и y_train
    X_train = df_balanced.drop(columns=['TARGET']).values
    y_train = df_balanced['TARGET'].values
    X_train = np.array(X_train, dtype=np.int64)  # Привести к числовому типу
    y_train = np.array(y_train, dtype=np.int64)  # Привести к числовому типу

    # Проверяем новое распределение
    print("Распределение после балансировки:\n", pd.Series(y_train).value_counts())
    # Конец балансировки --------------------------------------------------------------------------

    # === 5. СОЗДАНИЕ DATASET и DATALOADER ===
    X_test = np.array(X_test, dtype=np.int64)  # Привести к числовому типу
    y_test = np.array(y_test, dtype=np.int64)  # Привести к числовому типу

    train_dataset = CandlestickDataset(X_train, y_train)
    test_dataset = CandlestickDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, worker_init_fn=seed_worker)

    # === 6. ОБУЧЕНИЕ МОДЕЛИ С ОПТИМИЗАЦИЕЙ ПО P/L ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CandleLSTM(vocab_size=27, embedding_dim=8, hidden_dim=32, output_dim=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_pnl = float('-inf')  # Лучшая прибыль (изначально -∞)
    epoch_best_pnl = 0
    model_path = Path(fr"model\best_model_{counter}.pth")
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
        y_preds = []
        
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                y_pred = model(X_batch).squeeze().cpu().numpy()
                y_preds.extend(y_pred)

        # === Расчет P/L ===
        test_open_prices = df_fut['OPEN'].iloc[split:].values
        test_close_prices = df_fut['CLOSE'].iloc[split:].values
        pnl = calculate_pnl(y_preds, test_open_prices, test_close_prices)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {total_loss / len(train_loader):.5f}, "
            f"P/L: {pnl:.2f}, "
            f"Best P/L: {best_pnl:.2f}, "
            f"Epoch best P/L: {epoch_best_pnl}, "
            f"seed: {counter}"
        )

        # === Сохранение лучшей модели по P/L ===
        if pnl > best_pnl:
            best_pnl = pnl
            epochs_no_improve = 0
            epoch_best_pnl = epoch + 1
            torch.save(model.state_dict(), model_path)
            print(f"✅ Model saved with P/L: {best_pnl:.2f}")
        else:
            epochs_no_improve += 1

        # === Ранняя остановка ===
        if epochs_no_improve >= early_stop_epochs:
            print(f"🛑 Early stopping at epoch {epoch + 1}")
            break

    # === 7. ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ И ФИНАЛЬНЫЙ ТЕСТ ===
    print("\n🔹 Loading best model for final evaluation...")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_preds_final = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).squeeze().cpu().numpy()
            y_preds_final.extend(y_pred)

    final_pnl = calculate_pnl(y_preds_final, test_open_prices, test_close_prices)
    print(f"🏆 Final Test P/L: {final_pnl:.2f}")
