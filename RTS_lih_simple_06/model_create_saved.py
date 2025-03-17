import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import json
import os
# Импортируем балансировку и кодировку свечей
from data_processing import balance_classes, encode_candle, calculate_pnl

# === СОЗДАНИЕ НЕЙРОСЕТИ (LSTM) ===
class CandleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(CandleLSTM, self).__init__()

        # Embedding слой для кодов свечей
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM принимает объединенные фичи (embedding + volume)
        self.lstm = nn.LSTM(embedding_dim + 1, hidden_dim, batch_first=True)

        # Полносвязный слой для предсказания
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_candle, x_volume):
        # Преобразуем коды свечей в embedding
        x_candle = self.embedding(x_candle)

        # Объединяем свечи и объем (по оси признаков)
        x = torch.cat((x_candle, x_volume.unsqueeze(-1)), dim=-1)

        # Пропускаем через LSTM
        x, _ = self.lstm(x)

        # Полносвязный слой и сигмоида
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)

class CandlestickDataset(Dataset):
    def __init__(self, X_candle, X_volume, y):
        self.X_candle = torch.tensor(X_candle, dtype=torch.long)
        self.X_volume = torch.tensor(X_volume, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_candle)  # ✅ Исправлено

    def __getitem__(self, idx):
        return self.X_candle[idx], self.X_volume[idx], self.y[idx]
    
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

# === 1. ОПРЕДЕЛЕНИЯ ===
# Установка рабочей директории в папку, где находится файл скрипта
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Загрузка полного словаря для преобразования кода свечи в числовой формат
with open("code_full_int.json", "r") as f:
    code_to_int = json.load(f)

db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_day_full.db')

for counter in range(1, 101):
    set_seed(counter)  # Устанавливаем одинаковый seed

    # === 2. ЗАГРУЗКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ И ВАЛИДАЦИИ ===
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

    # Создание кодов свечей по Лиховидову
    df_fut['CANDLE_CODE'] = df_fut.apply(encode_candle, axis=1)

    # === 3. ПОДГОТОВКА ДАННЫХ ===
    # Преобразуем свечные коды в числовой формат (список уникальных кодов)
    df_fut['CANDLE_INT'] = df_fut['CANDLE_CODE'].map(code_to_int)

    # Создание колонки направления.
    df_fut['DIRECTION'] = (df_fut['CLOSE'] > df_fut['OPEN']).astype(int)

    # Создание колонок с признаками 'CANDLE_INT' за 20 предыдущих свечей
    for i in range(1, 21):
        df_fut[f'CI_{i}'] = df_fut['CANDLE_INT'].shift(i).astype('Int64')

    # Создание колонок с объемом за 20 предыдущих свечей
    for i in range(1, 21):
        df_fut[f'VOL_{i}'] = df_fut['VOLUME'].shift(i).astype('Int64')

    df_fut = df_fut.dropna().reset_index(drop=True)

    # Создание дата сетов
    X_candle = df_fut[[f'CI_{i}' for i in range(1, 21)]].values
    X_volume = df_fut[[f'VOL_{i}' for i in range(1, 21)]].values
    y = df_fut['DIRECTION']
    X_candle, X_volume, y = np.array(X_candle), np.array(X_volume), np.array(y)

    # Нормализация объема
    scaler = StandardScaler()
    # # Нормализация относительно всего дата-сета
    # X_volume = scaler.fit_transform(X_volume)
    # Нормализация по окну из 20 значений.
    X_volume = np.array([scaler.fit_transform(row.reshape(-1, 1)).flatten() for row in X_volume])

    # Разделение на train/test
    split = int(0.85 * len(y))
    X_train_candle, X_train_volume, y_train = X_candle[:split], X_volume[:split], y[:split]
    X_test_candle, X_test_volume, y_test = X_candle[split:], X_volume[split:], y[split:]

    # === 4. Балансировка классов ===
    X_train_candle, X_train_volume, y_train = balance_classes(X_train_candle, X_train_volume, y_train)

    # === 5. СОЗДАНИЕ DATASET и DATALOADER ===
    X_train_candle = np.array(X_train_candle, dtype=np.int64)  # Привести к числовому типу
    X_train_volume = np.array(X_train_volume, dtype=np.float32)  # Привести к числовому типу
    y_train = np.array(y_train, dtype=np.int64)  # Привести к числовому типу
    X_test_candle = np.array(X_test_candle, dtype=np.int64)  # Привести к числовому типу
    X_test_volume = np.array(X_test_volume, dtype=np.float32)  # Привести к числовому типу
    y_test = np.array(y_test, dtype=np.int64)  # Привести к числовому типу

    train_dataset = CandlestickDataset(X_train_candle, X_train_volume, y_train)
    test_dataset = CandlestickDataset(X_test_candle, X_test_volume, y_test)

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
        for X_candle_batch, X_volume_batch, y_batch in train_loader:
            X_candle_batch, X_volume_batch, y_batch = (
                X_candle_batch.to(device),
                X_volume_batch.to(device),
                y_batch.to(device)
            )

            optimizer.zero_grad()
            y_pred = model(X_candle_batch, X_volume_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # === Проверка на тесте после каждой эпохи ===
        model.eval()
        y_preds = []

        with torch.no_grad():
            for X_candle_batch, X_volume_batch, _ in test_loader:
                X_candle_batch, X_volume_batch = X_candle_batch.to(device), X_volume_batch.to(device)
                y_pred = model(X_candle_batch, X_volume_batch).squeeze().cpu().numpy()
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
        for X_candle_batch, X_volume_batch, _ in test_loader:
            X_candle_batch, X_volume_batch = X_candle_batch.to(device), X_volume_batch.to(device)
            y_pred = model(X_candle_batch, X_volume_batch).squeeze().cpu().numpy()
            y_preds_final.extend(y_pred)

    final_pnl = calculate_pnl(y_preds_final, test_open_prices, test_close_prices)
    print(f"🏆 Final Test P/L: {final_pnl:.2f}")
