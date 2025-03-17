import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import os

# Импортируем балансировку и кодировку свечей
from data_processing import balance_classes, data_prepare, calculate_pnl

# === 1. СОЗДАНИЕ НЕЙРОСЕТИ (LSTM) ===
class CandleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, day_vocab_size, day_embedding_dim, hidden_dim, output_dim):
        super(CandleLSTM, self).__init__()

        # Embedding слой для кодов свечей
        self.embedding_candle = nn.Embedding(vocab_size, embedding_dim)
        
        # Embedding слой для дня недели
        self.embedding_day = nn.Embedding(day_vocab_size, day_embedding_dim)

        # LSTM принимает объединенные фичи (embedding свечей + объем + embedding дня недели)
        self.lstm = nn.LSTM(embedding_dim + 1 + day_embedding_dim, hidden_dim, batch_first=True)

        # Полносвязный слой для предсказания
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_candle, x_volume, x_day):
        # Преобразуем коды свечей и день недели в embeddings
        x_candle = self.embedding_candle(x_candle)
        x_day = self.embedding_day(x_day)

        # Объединяем свечи, объем и день недели
        x = torch.cat((x_candle, x_volume.unsqueeze(-1), x_day), dim=-1)

        # Пропускаем через LSTM
        x, _ = self.lstm(x)

        # Полносвязный слой и сигмоида
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)

# === 2. СОЗДАНИЕ DATASET ===
class CandlestickDataset(Dataset):
    def __init__(self, X_candle, X_volume, X_day, y):
        self.X_candle = torch.tensor(X_candle, dtype=torch.long)
        self.X_volume = torch.tensor(X_volume, dtype=torch.float32)
        self.X_day = torch.tensor(X_day, dtype=torch.long)  # Дни недели как long (категориальные)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_candle)

    def __getitem__(self, idx):
        return self.X_candle[idx], self.X_volume[idx], self.X_day[idx], self.y[idx]

# === 3. ФИКСАЦИЯ СЛУЧАЙНЫХ ЧИСЕЛ ДЛЯ ДЕТЕРМИНИРОВАННОСТИ ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === 4. ЗАГРУЗКА ДАННЫХ ===
db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_day_full.db')

# Установка рабочей директории в папку, где находится файл скрипта
script_dir = Path(__file__).parent
os.chdir(script_dir)

for counter in range(1, 101):
    set_seed(counter)

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

    df_fut = data_prepare(df_fut)

    # Создание дата сетов
    X_candle = df_fut[[f'CI_{i}' for i in range(1, 21)]].values
    X_volume = df_fut[[f'VOL_{i}' for i in range(1, 21)]].values
    X_day = df_fut[[f'DAY_W_{i}' for i in range(1, 21)]].values
    y = df_fut['DIRECTION']
    
    X_candle, X_volume, X_day, y = map(np.array, [X_candle, X_volume, X_day, y])

    # Разделение на train/test
    split = int(0.85 * len(y))
    X_train_candle, X_train_volume, X_train_day, y_train = X_candle[:split], X_volume[:split], X_day[:split], y[:split]
    # X_train_candle, X_train_volume, X_train_day, y_train = map(lambda x: x[:split], [X_candle, X_volume, X_day, y])
    X_test_candle, X_test_volume, X_test_day, y_test = X_candle[split:], X_volume[split:], X_day[split:], y[split:]

    # Балансировка классов
    X_train_candle, X_train_volume, X_train_day, y_train = balance_classes(X_train_candle, X_train_volume, X_train_day, y_train)

    # Создание dataset и data loader
    train_dataset = CandlestickDataset(X_train_candle, X_train_volume, X_train_day, y_train)
    test_dataset = CandlestickDataset(X_test_candle, X_test_volume, X_test_day, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # === 5. ОБУЧЕНИЕ МОДЕЛИ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CandleLSTM(vocab_size=27, embedding_dim=8, day_vocab_size=7, day_embedding_dim=4, hidden_dim=32, output_dim=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_pnl = float('-inf')
    epoch_best_pnl = 0
    model_path = Path(fr"model\best_model_{counter}.pth")
    early_stop_epochs = 200
    epochs_no_improve = 0

    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_candle_batch, X_volume_batch, X_day_batch, y_batch in train_loader:
            X_candle_batch, X_volume_batch, X_day_batch, y_batch = (
                X_candle_batch.to(device),
                X_volume_batch.to(device),
                X_day_batch.to(device),
                y_batch.to(device)
            )

            optimizer.zero_grad()
            y_pred = model(X_candle_batch, X_volume_batch, X_day_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # === Проверка на тесте ===
        model.eval()
        y_preds = []
        with torch.no_grad():
            for X_candle_batch, X_volume_batch, X_day_batch, _ in test_loader:
                X_candle_batch, X_volume_batch, X_day_batch = X_candle_batch.to(device), X_volume_batch.to(device), X_day_batch.to(device)
                y_pred = model(X_candle_batch, X_volume_batch, X_day_batch).squeeze().cpu().numpy()
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

        if pnl > best_pnl:
            best_pnl = pnl
            epochs_no_improve = 0
            epoch_best_pnl = epoch + 1
            torch.save(model.state_dict(), model_path)
            print(f"✅ Model saved with P/L: {best_pnl:.2f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_epochs:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break

    # === 7. ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ И ФИНАЛЬНЫЙ ТЕСТ ===
    print("\n🔹 Loading best model for final evaluation...")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_preds_final = []
    with torch.no_grad():
        for X_candle_batch, X_volume_batch, X_day_batch, _ in test_loader:
            X_candle_batch, X_volume_batch, X_day_batch = X_candle_batch.to(device), X_volume_batch.to(device), X_day_batch.to(device)
            y_pred = model(X_candle_batch, X_volume_batch, X_day_batch).squeeze().cpu().numpy()
            y_preds_final.extend(y_pred)

    final_pnl = calculate_pnl(y_preds_final, test_open_prices, test_close_prices)
    print(f"🏆 Финальный тест P/L: {final_pnl:.2f}\n")
