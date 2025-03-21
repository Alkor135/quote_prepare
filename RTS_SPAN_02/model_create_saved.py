import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import os
from data_read import data_load
from data_processing import balance_classes, calculate_pnl
import shutil
import sys
sys.dont_write_bytecode = True

# === ФИКСАЦИЯ СЛУЧАЙНЫХ ЧИСЕЛ ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === НЕЙРОСЕТЬ (LSTM) ===
class CandleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CandleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        if len(x.shape) == 2:  # Если (batch_size, hidden_dim), то просто подаем в FC
            x = self.fc(x)
        else:  # Если (batch_size, seq_len, hidden_dim), берем последний таймстеп
            x = self.fc(x[:, -1, :])
        return self.sigmoid(x)


# === DATASET ===
class CandlestickDataset(Dataset):
    def __init__(self, *X, y):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X]
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return tuple(x[idx] for x in self.X), self.y[idx]


# === ЗАГРУЗКА ДАННЫХ ===
script_dir = Path(__file__).parent
os.chdir(script_dir)

db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day_2014.db')
df = data_load(db_path, '2014-01-01', '2024-01-01')
# print(df)

for counter in range(1, 101):
    # Удаляем папку __pycache__ (если она была создана)
    shutil.rmtree('__pycache__', ignore_errors=True)
    
    set_seed(counter)
    df_fut = df.copy()

    X_0 = df_fut[[f'mp0_{i}' for i in range(1, 11)]].values
    X_1 = df_fut[[f'mp1_{i}' for i in range(1, 11)]].values
    X_2 = df_fut[[f'mp2_{i}' for i in range(1, 11)]].values
    X_3 = df_fut[[f'mp3_{i}' for i in range(1, 11)]].values
    X_4 = df_fut[[f'mp4_{i}' for i in range(1, 11)]].values
    X_5 = df_fut[[f'mp5_{i}' for i in range(1, 11)]].values
    X_6 = df_fut[[f'mp6_{i}' for i in range(1, 11)]].values
    y = df_fut['DIRECTION'].values

    split = int(0.85 * len(y))
    X_train_0, X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_train_6, y_train = (
        X_0[:split], X_1[:split], X_2[:split], X_3[:split], X_4[:split], X_5[:split], X_6[:split], y[:split]
        )
    X_test_0, X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6, y_test = (
        X_0[split:], X_1[split:], X_2[split:], X_3[split:], X_4[split:], X_5[split:], X_6[split:], y[split:]
        )

    X_train_0, X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_train_6, y_train = balance_classes(
        X_train_0, X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_train_6, y_train
        )

    train_dataset = CandlestickDataset(X_train_0, X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_train_6, y=y_train)
    test_dataset = CandlestickDataset(X_test_0, X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6, y=y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # === ОБУЧЕНИЕ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandleLSTM(input_dim=70, hidden_dim=128, output_dim=1).to(device)
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
        for X_batch, y_batch in train_loader:
            X_batch = torch.cat(X_batch, dim=-1).to(device)  # Соединение по последнему измерению
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()

            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # === ТЕСТ ===
        model.eval()
        y_preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = torch.cat(X_batch, dim=-1).to(device)
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

    # === ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ И ФИНАЛЬНЫЙ ТЕСТ ===
    print("\n🔹 Loading best model for final evaluation...")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_preds_final = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = torch.cat(X_batch, dim=-1).to(device)
            y_pred = model(X_batch).squeeze().cpu().numpy()
            y_preds_final.extend(y_pred)

    final_pnl = calculate_pnl(y_preds_final, test_open_prices, test_close_prices)
    print(f"🏆 Финальный тест P/L: {final_pnl:.2f}\n")
