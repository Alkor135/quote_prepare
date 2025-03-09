import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# === 1. ФИКСАЦИЯ СЛУЧАЙНЫХ ЧИСЕЛ ДЛЯ ДЕТЕРМИНИРОВАННОСТИ ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_var = 79
set_seed(seed_var)  # Устанавливаем одинаковый seed

# === 2. ЗАГРУЗКА ДАННЫХ ===
db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_day_full.db')
with sqlite3.connect(db_path) as conn:
    df_fut = pd.read_sql_query(
        """
        SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME 
        FROM Day 
        ORDER BY TRADEDATE
        """,
        conn
    )

# === 3. ФУНКЦИЯ КОДИРОВАНИЯ СВЕЧЕЙ (ЛИХОВИДОВ) ===
def encode_candle(row):
    open_, low, high, close = row['OPEN'], row['LOW'], row['HIGH'], row['CLOSE']

    direction = 1 if close > open_ else (0 if close < open_ else 2)
    upper_shadow = high - max(open_, close)
    lower_shadow = min(open_, close) - low
    body = abs(close - open_)

    def classify_shadow(shadow, body):
        return 0 if shadow < 0.1 * body else (1 if shadow < 0.5 * body else 2)

    return f"{direction}{classify_shadow(upper_shadow, body)}{classify_shadow(lower_shadow, body)}"

df_fut['CANDLE_CODE'] = df_fut.apply(encode_candle, axis=1)

# === 4. ПОДГОТОВКА ДАННЫХ ===
unique_codes = sorted(df_fut['CANDLE_CODE'].unique())
code_to_int = {code: i for i, code in enumerate(unique_codes)}
df_fut['CANDLE_INT'] = df_fut['CANDLE_CODE'].map(code_to_int)

window_size = 20
predict_offset = 1

X, y = [], []
for i in range(len(df_fut) - window_size - predict_offset):
    X.append(df_fut['CANDLE_INT'].iloc[i:i + window_size].values)
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
# print(X_train)  # Проверка что перемешивание не испортит результат. Фичи сформированы.

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=seed_worker)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, worker_init_fn=seed_worker)


# === 5. СОЗДАНИЕ НЕЙРОСЕТИ (LSTM) ===
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

# === 6. ОБУЧЕНИЕ МОДЕЛИ С СОХРАНЕНИЕМ ЛУЧШЕЙ ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CandleLSTM(vocab_size=len(unique_codes), embedding_dim=8, hidden_dim=32, output_dim=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_accuracy = 0
epoch_best_accuracy = 0
model_path = Path(r"best_model_graph_RTS_bal_01.pth")
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
        f"seed: {seed_var}"
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

# === 7. ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ И ТЕСТ ===
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
