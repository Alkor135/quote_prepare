import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import os
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
    def __init__(self, vocab_size, embedding_dim, day_vocab_size, day_embedding_dim, 
                 dd_vocab_size, dd_embedding_dim, hidden_dim, output_dim):
        super(CandleLSTM, self).__init__()
        self.embedding_candle = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_day = nn.Embedding(day_vocab_size, day_embedding_dim)
        self.embedding_dd = nn.Embedding(dd_vocab_size, dd_embedding_dim)
        input_dim = embedding_dim + 1 + day_embedding_dim + dd_embedding_dim + 5
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_span):
        x = torch.cat((x_span.unsqueeze(-1)), dim=-1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)

# === DATASET ===
class CandlestickDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === ЗАГРУЗКА ДАННЫХ ===
script_dir = Path(__file__).parent
os.chdir(script_dir)
# Загружаем файл csv в DF, дату делаем типом datetime
df = pd.read_csv(r'span_nn_prepare.csv', parse_dates=['TRADEDATE'])

for counter in range(1, 101):
    # Удаляем папку __pycache__ (если она была создана)
    shutil.rmtree('__pycache__', ignore_errors=True)

    set_seed(counter)
    df_fut = df.query("'2014-01-01' <= TRADEDATE <= '2024-01-01'")

    X_span = df_fut[[i for i in range(-25000, 25001, 2500)]].values
    y = df_fut['DIRECTION'].values  # Преобразуем сразу в numpy

    split = int(0.85 * len(y))
    X_train, y_train = (X_span[:split], y[:split])
    X_test, y_test = (X_span[split:], y[split:])

    X_train, y_train = balance_classes(X_train, y_train)

    train_dataset = CandlestickDataset(X_train, y_train)
    test_dataset = CandlestickDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # === ОБУЧЕНИЕ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandleLSTM(27, 8, 7, 4, 104, 4, 180, 1).to(device)
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
            X_batch, y_batch = (X_batch.to(device), y_batch.to(device))

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
                y_pred = model(X_batch.to(device)).squeeze().cpu().numpy()
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
        for X_batch, _ in test_loader:
            X_batch = (X_batch.to(device))
            y_pred = model(X_batch).squeeze().cpu().numpy()
            y_preds_final.extend(y_pred)

    final_pnl = calculate_pnl(y_preds_final, test_open_prices, test_close_prices)
    print(f"🏆 Финальный тест P/L: {final_pnl:.2f}\n")
