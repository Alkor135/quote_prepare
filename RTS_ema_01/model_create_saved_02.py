import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random  # Импортируем random!
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from data_read import data_load, balance_classes
import os
import shutil

# === ФИКСАЦИЯ СЛУЧАЙНЫХ ЧИСЕЛ ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === Основные параметры ===
SEQUENCE_LENGTH = 20
# INPUT_SIZE = SEQUENCE_LENGTH
INPUT_SIZE = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100

# === 1. Подготовка данных ===
def prepare_data(df_fut):
    X = df_fut[[f"ed_{i}" for i in range(1, 21)]].values.astype(np.float32)
    y = df_fut["DIRECTION"].values.astype(np.int64)

    train_size = int(0.85 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    X_train_bal, y_train_bal = balance_classes(X_train, y_train)

    print(f"X_train_bal shape после балансировки: {X_train_bal.shape}")

    num_samples = X_train_bal.shape[0]
    
    # 🔥 Убедимся, что размерность входных данных правильная
    X_train_bal = torch.tensor(X_train_bal).clone().detach().reshape(num_samples, SEQUENCE_LENGTH, 1)  
    y_train_bal = torch.tensor(y_train_bal)

    X_val = torch.tensor(X_val).clone().detach().reshape(-1, SEQUENCE_LENGTH, 1)  
    y_val = torch.tensor(y_val)

    train_dataset = TensorDataset(X_train_bal, y_train_bal)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    df_val = df_fut.iloc[train_size:]  # 🔥 Сохраняем df для PnL

    return train_loader, val_loader, df_val


# === 2. Определение модели LSTM ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output

# === 3. Функция тренировки ===
def train_model(train_loader, val_loader, df_val, counter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_pnl = float('-inf')  # Лучший PnL
    epoch_best_pnl = 0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct, total = 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        train_acc = correct / total
        val_pnl = evaluate_model_pnl(model, val_loader, df_val, device)  # 🔥 Новый расчет PnL

        print(f"Epoch {epoch+1}/{EPOCHS}, "
              f"Loss: {total_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Val PnL: {val_pnl:.2f}, "
              f"Best PnL: {best_pnl:.2f}, "
              f"Epoch best P/L: {epoch_best_pnl}, "
              f"seed: {counter}"
              )

        # Сохранение модели с наилучшим PnL
        if val_pnl > best_pnl:
            best_pnl = val_pnl
            epochs_no_improve = 0
            epoch_best_pnl = epoch + 1
            torch.save(model.state_dict(), Path(fr"model/best_lstm_model_{counter}.pth"))
            print("🔥 Model saved based on PnL!")
        else:
            epochs_no_improve += 1

# === 4. Оценка на валидации ===
def evaluate_model_pnl(model, val_loader, df_val, device):
    model.eval()
    total_pnl = 0.0

    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(val_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)

            # Берем реальные OPEN и CLOSE для текущей пачки данных
            batch_start = i * BATCH_SIZE
            batch_end = batch_start + len(y_batch)
            open_prices = df_val.iloc[batch_start:batch_end]["OPEN"].values
            close_prices = df_val.iloc[batch_start:batch_end]["CLOSE"].values

            # Рассчитываем PnL для каждого примера в batch
            for j in range(len(y_batch)):
                price_diff = abs(close_prices[j] - open_prices[j])
                if predicted[j].item() == y_batch[j].item():
                    total_pnl += price_diff  # Плюсуем, если прогноз правильный
                else:
                    total_pnl -= price_diff  # Минусуем, если ошиблись

    return total_pnl

# === 5. Запуск обучения ===
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Удаляем папку __pycache__ (если она была создана)
    shutil.rmtree('__pycache__', ignore_errors=True)

    db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day_2014.db')
    df_fut = data_load(db_path, '2014-01-01', '2024-01-01')

    train_loader, val_loader, df_val = prepare_data(df_fut)

    for counter in range(1, 101):
        # Удаляем папку __pycache__ (если она была создана)
        shutil.rmtree('__pycache__', ignore_errors=True)

        set_seed(counter)  # Фиксируем случайные числа
        train_model(train_loader, val_loader, df_val, counter)  # 🔥 Передаем df_val для расчета
        print()
