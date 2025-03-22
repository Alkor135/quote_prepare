import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
from data_read import data_load, balance_classes
import os

# === Параметры ===
SEQUENCE_LENGTH = 20  # Количество свечей в истории
INPUT_SIZE = SEQUENCE_LENGTH  # Кол-во фичей (ed_1 - ed_20)
HIDDEN_SIZE = 64  # Размер скрытого слоя LSTM
NUM_LAYERS = 2  # Число LSTM-слоёв
OUTPUT_SIZE = 2  # Два класса: 0 (падение) и 1 (рост)
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
MODEL_PATH = Path(r"model/best_lstm_model.pth")

#     return train_loader, val_loader
def prepare_data(df_fut):
    X = df_fut[[f"ed_{i}" for i in range(1, 21)]].values.astype(np.float32)
    y = df_fut["DIRECTION"].values.astype(np.int64)

    # Разделение на train/val
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Балансировка train-данных
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)

    # Проверяем размерности
    print(f"X_train_bal shape после балансировки: {X_train_bal.shape}")  # Должно быть (samples, 20)

    # Преобразование в тензоры
    num_samples = X_train_bal.shape[0]  # Количество примеров после балансировки
    X_train_bal = torch.tensor(X_train_bal).reshape(num_samples, 1, INPUT_SIZE)  # (samples, 1, 20)
    y_train_bal = torch.tensor(y_train_bal)

    X_val = torch.tensor(X_val).reshape(-1, 1, INPUT_SIZE)  # (samples, 1, 20)
    y_val = torch.tensor(y_val)

    # Создание датасетов
    train_dataset = TensorDataset(X_train_bal, y_train_bal)
    val_dataset = TensorDataset(X_val, y_val)

    # DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

# === 2. Определение модели LSTM ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Берём последний временной шаг
        output = self.fc(last_hidden)
        return output

# === 3. Функция тренировки ===
def train_model(train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

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
        val_acc = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("🔥 Model saved!")

# === 4. Оценка на валидации ===
def evaluate_model(model, val_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total

# === 5. Запуск обучения ===
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day_2014.db')
    df_fut = data_load(db_path, '2014-01-01', '2024-01-01')

    train_loader, val_loader = prepare_data(df_fut)
    train_model(train_loader, val_loader)
