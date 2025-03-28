import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import os
from data_read import data_load, balance_classes
import shutil
import sys
sys.dont_write_bytecode = True

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


# === Установка фиксированного seed для воспроизводимости ===
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

# === 2. ЗАГРУЗКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ И ВАЛИДАЦИИ ===
db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_day_2014.db')
df = data_load(db_path, '2014-01-01', '2024-01-01')

for counter in range(1, 101):
    # Удаляем папку __pycache__ (если она была создана)
    shutil.rmtree('__pycache__', ignore_errors=True)

    set_seed(counter)  # Устанавливаем одинаковый seed

    df_fut = df.copy()

    # Создание дата сетов
    X = df_fut[[f'CI_{i}' for i in range(1, 21)]].values
    y = df_fut['DIRECTION']
    X, y = np.array(X), np.array(y)

    # Разделение на train/test
    split = int(0.85 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # === 4. Балансировка классов === 
    X_train, y_train = balance_classes(X_train, y_train)

    # === 5. СОЗДАНИЕ DATASET и DATALOADER ===
    X_test = np.array(X_test, dtype=np.int64)  # Привести к числовому типу
    y_test = np.array(y_test, dtype=np.int64)  # Привести к числовому типу

    train_dataset = CandlestickDataset(X_train, y_train)
    test_dataset = CandlestickDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, worker_init_fn=seed_worker
        )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, worker_init_fn=seed_worker
        )

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
