import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import os
from itertools import product
from data_read import data_load, balance_classes
import shutil
import sys
sys.dont_write_bytecode = True

# === Установка фиксированного seed ===
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Устанавливаем фиксированный seed
SEED = 42
set_seed(SEED)

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
    def __init__(self, X, y, bodies, avg_bodies):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.bodies = torch.tensor(bodies, dtype=torch.float32)
        self.avg_bodies = torch.tensor(avg_bodies, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.bodies[idx], self.avg_bodies[idx]

# === Кастомная функция потерь с учетом размера свечи ===
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, candle_bodies, avg_body_20):
        y_pred_classes = (y_pred > 0.5).float()
        correct = (y_pred_classes == y_true).float()
        large_body = (candle_bodies > avg_body_20).float()

        weights = correct * (1 + large_body) + (1 - correct) * (1 + large_body)
        weights = torch.clamp(weights, min=0.1)  

        loss = nn.BCELoss(reduction="none")(y_pred, y_true)
        weighted_loss = loss * weights

        return weighted_loss.mean()

# === Функция расчета P/L ===
def calculate_pnl(y_preds, open_prices, close_prices):
    pnl = 0
    for i in range(len(y_preds)):
        if y_preds[i] > 0.5:  
            pnl += close_prices[i] - open_prices[i]
        else:  
            pnl += open_prices[i] - close_prices[i]
    return pnl  

# === Основная логика ===
script_dir = Path(__file__).parent
os.chdir(script_dir)

db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day_2014.db')
df = data_load(db_path, '2014-01-01', '2024-01-01')

# === Гиперпараметры для перебора ===
param_grid = {
    'embedding_dim': [8, 16, 32],
    'hidden_dim': [32, 64, 128],
    'batch_size': [16, 32, 64],
    'learning_rate': [0.001, 0.0005]
}

# Генерация всех комбинаций гиперпараметров
param_combinations = list(product(
    param_grid['embedding_dim'],
    param_grid['hidden_dim'],
    param_grid['batch_size'],
    param_grid['learning_rate']
))

for params in param_combinations:
    embedding_dim, hidden_dim, batch_size, learning_rate = params
    print(f"Текущая комбинация гиперпараметров: embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, batch_size={batch_size}, learning_rate={learning_rate}")

    shutil.rmtree('__pycache__', ignore_errors=True)

    df_fut = df.copy()

    X = df_fut[[f'CI_{i}' for i in range(1, 21)]].values
    y = df_fut['DIRECTION']
    bodies = df_fut['BODY']
    avg_bodies = df_fut['BODY_AVG']

    X, y, bodies, avg_bodies = map(np.array, [X, y, bodies, avg_bodies])

    split = int(0.85 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    bodies_train, bodies_test = bodies[:split], bodies[split:]
    avg_bodies_train, avg_bodies_test = avg_bodies[:split], avg_bodies[split:]

    X_train, y_train, bodies_train, avg_bodies_train = balance_classes(
        X_train, y_train, bodies_train, avg_bodies_train
    )

    train_dataset = CandlestickDataset(X_train, y_train, bodies_train, avg_bodies_train)
    test_dataset = CandlestickDataset(X_test, y_test, bodies_test, avg_bodies_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CandleLSTM(vocab_size=27, embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=1).to(device)
    criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_pnl = float('-inf')
    model_path = Path(fr"model_gip\best_model_{embedding_dim}_{hidden_dim}_{batch_size}_{learning_rate}.pth")

    epochs = 200
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch, bodies_batch, avg_bodies_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            bodies_batch, avg_bodies_batch = bodies_batch.to(device), avg_bodies_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch, bodies_batch, avg_bodies_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_preds = []
        with torch.no_grad():
            for X_batch, _, _, _ in test_loader:
                X_batch = X_batch.to(device)
                y_pred = model(X_batch).squeeze().cpu().numpy()
                y_preds.extend(y_pred)

        test_open_prices = df_fut['OPEN'].iloc[split:].values
        test_close_prices = df_fut['CLOSE'].iloc[split:].values
        pnl = calculate_pnl(y_preds, test_open_prices, test_close_prices)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {total_loss / len(train_loader):.5f}, "
            f"P/L: {pnl:.2f}, "
            f"Best P/L: {best_pnl:.2f}"
        )

        if pnl > best_pnl:
            best_pnl = pnl
            torch.save(model.state_dict(), model_path)

    print(f"🏆 Лучшая модель для текущей комбинации сохранена: {model_path}")
    print()
# === КОНЕЦ ===
