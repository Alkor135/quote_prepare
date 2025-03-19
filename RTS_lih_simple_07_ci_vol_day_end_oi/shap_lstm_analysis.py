import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import os
import shap
import matplotlib.pyplot as plt

from data_read import data_load
from data_processing import balance_classes, calculate_pnl

# === 1. СОЗДАНИЕ НЕЙРОСЕТИ (LSTM) ===
class CandleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, day_vocab_size, day_embedding_dim, 
                 dd_vocab_size, dd_embedding_dim, hidden_dim, output_dim):
        super(CandleLSTM, self).__init__()

        self.embedding_candle = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_day = nn.Embedding(day_vocab_size, day_embedding_dim)
        self.embedding_dd = nn.Embedding(dd_vocab_size, dd_embedding_dim)

        input_dim = embedding_dim + 1 + day_embedding_dim + dd_embedding_dim + 1  # 1 для X_io
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_candle, x_volume, x_day, x_dd, x_io):
        x_candle = self.embedding_candle(x_candle)
        x_day = self.embedding_day(x_day)
        x_dd = self.embedding_dd(x_dd)

        x = torch.cat((x_candle, x_volume.unsqueeze(-1), x_day, x_dd, x_io.unsqueeze(-1)), dim=-1)

        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)

# === 2. СОЗДАНИЕ DATASET ===
class CandlestickDataset(Dataset):
    def __init__(self, X_candle, X_volume, X_day, X_dd, X_io, y):
        self.X_candle = torch.tensor(X_candle, dtype=torch.long)
        self.X_volume = torch.tensor(X_volume, dtype=torch.float32)
        self.X_day = torch.tensor(X_day, dtype=torch.long)
        self.X_dd = torch.tensor(X_dd, dtype=torch.long)
        self.X_io = torch.tensor(X_io, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_candle[idx], self.X_volume[idx], self.X_day[idx], self.X_dd[idx], self.X_io[idx], self.y[idx]

# === 3. ФИКСАЦИЯ СЛУЧАЙНЫХ ЧИСЕЛ ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === 4. ЗАГРУЗКА ДАННЫХ ===
db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day_2014.db')

script_dir = Path(__file__).parent
os.chdir(script_dir)

set_seed(42)

start_date = '2014-01-01'
end_date = '2024-01-01'
df_fut = data_load(db_path, start_date, end_date)

X_candle = df_fut[[f'CI_{i}' for i in range(1, 21)]].values
X_volume = df_fut[[f'VOL_{i}' for i in range(1, 21)]].values
X_day = df_fut[[f'DAY_W_{i}' for i in range(1, 21)]].values
X_dd = df_fut[[f'DD_{i}' for i in range(1, 21)]].values
X_io = df_fut[[f'IO_{i}' for i in range(1, 21)]].values
y = df_fut['DIRECTION'].values

split = int(0.85 * len(y))
X_train_candle, X_train_volume, X_train_day, X_train_dd, X_train_io, y_train = map(lambda x: x[:split], [X_candle, X_volume, X_day, X_dd, X_io, y])
X_test_candle, X_test_volume, X_test_day, X_test_dd, X_test_io, y_test = map(lambda x: x[split:], [X_candle, X_volume, X_day, X_dd, X_io, y])

X_train_candle, X_train_volume, X_train_day, X_train_dd, X_train_io, y_train = balance_classes(
    X_train_candle, X_train_volume, X_train_day, X_train_dd, X_train_io, y_train
)

train_dataset = CandlestickDataset(X_train_candle, X_train_volume, X_train_day, X_train_dd, X_train_io, y_train)
test_dataset = CandlestickDataset(X_test_candle, X_test_volume, X_test_day, X_test_dd, X_test_io, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# === 5. ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CandleLSTM(
    vocab_size=27, embedding_dim=8, 
    day_vocab_size=7, day_embedding_dim=4, 
    dd_vocab_size=104, dd_embedding_dim=4,  
    hidden_dim=32, output_dim=1
).to(device)

model_path = Path(r"model\best_model_42.pth")
model.load_state_dict(torch.load(model_path))
model.eval()

# === 6. SHAP АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ ===
batch = next(iter(test_loader))
X_candle_batch, X_volume_batch, X_day_batch, X_dd_batch, X_io_batch, _ = batch

X_candle_batch = X_candle_batch.to(device)
X_volume_batch = X_volume_batch.to(device)
X_day_batch = X_day_batch.to(device)
X_dd_batch = X_dd_batch.to(device)
X_io_batch = X_io_batch.to(device)

# Убедимся, что все тензоры имеют размерность (batch_size, sequence_length, feature_dim)
X_candle_batch = X_candle_batch.unsqueeze(-1)  # (batch_size, sequence_length, 1)
X_day_batch = X_day_batch.unsqueeze(-1)        # (batch_size, sequence_length, 1)
X_dd_batch = X_dd_batch.unsqueeze(-1)          # (batch_size, sequence_length, 1)

# Объединяем все входные признаки в единый тензор
X_combined = torch.cat((X_candle_batch, X_volume_batch.unsqueeze(-1),
                        X_day_batch, X_dd_batch, X_io_batch.unsqueeze(-1)), dim=-1)

# Преобразуем тензор X_combined в правильную форму для SHAP
X_combined_np = X_combined.cpu().numpy().reshape(X_combined.shape[0], -1)  # (batch_size, seq_len * feature_dim)
masker = shap.maskers.Independent(X_combined_np)


# Оборачиваем модель в функцию, которая принимает массив и возвращает предсказания
def model_wrapper(x_np):
    x = torch.tensor(x_np, dtype=torch.float32, device=device)  # Преобразуем в тензор
    batch_size, _ = x.shape

    # Восстанавливаем оригинальную размерность (batch_size, seq_len, feature_dim)
    x = x.view(batch_size, 20, 5)

    # Разбиваем по фичам (на основе исходного порядка объединения данных)
    x_candle, x_volume, x_day, x_dd, x_io = torch.split(x, [1, 1, 1, 1, 1], dim=-1)

    # Преобразуем в правильные форматы для модели
    x_candle = x_candle.squeeze(-1).long()
    x_day = x_day.squeeze(-1).long()
    x_dd = x_dd.squeeze(-1).long()
    x_volume = x_volume.squeeze(-1)
    x_io = x_io.squeeze(-1)

    # Прогоняем через модель
    with torch.no_grad():
        return model(x_candle, x_volume, x_day, x_dd, x_io).cpu().numpy()

# Используем SHAP Masker для правильного формирования входных данных
masker = shap.maskers.Independent(X_combined_np)

# Создаем SHAP Explainer
explainer = shap.Explainer(model_wrapper, masker)

# Вычисляем значения SHAP
shap_values = explainer(X_combined_np, max_evals=2000)

# Визуализация
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_combined_np)
plt.show()

feature_names = (
    [f"CI_{i}" for i in range(1, 21)] +  # 0-19
    [f"VOL_{i}" for i in range(1, 21)] +  # 20-39
    [f"DAY_W_{i}" for i in range(1, 21)] +  # 40-59
    [f"DD_{i}" for i in range(1, 21)] +  # 60-79
    [f"IO_{i}" for i in range(1, 21)]  # 80-99
)

# Усредняем абсолютные значения SHAP-важностей по батчу
shap_importance = np.abs(shap_values.values).mean(axis=0)

# Создаем список (Feature №, Название, SHAP-важность)
feature_ranking = sorted(
    enumerate(zip(feature_names, shap_importance)),
    key=lambda x: x[1][1],  # сортируем по SHAP-важности
    reverse=True
)

# Выводим отсортированный список
for rank, (feature_idx, (feature_name, importance)) in enumerate(feature_ranking, 1):
    print(f"{rank}. Feature {feature_idx}: {feature_name} (SHAP: {importance:.4f})")
