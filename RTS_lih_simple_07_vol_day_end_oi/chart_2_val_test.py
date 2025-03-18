"""
Сохранение 2 графиков (валидационного и тестового) в файл.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
from data_read import data_load


# === ОПРЕДЕЛЕНИЕ МОДЕЛИ LSTM (ДОЛЖНА СОВПАДАТЬ С ОБУЧЕННОЙ) ===
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


# === РАСЧЁТ РЕЗУЛЬТАТОВ ПОСЛЕ ПРОГНОЗА ===
def calculate_result(row):
    if pd.isna(row["PREDICTION"]):  # Если NaN 
        return 0  # Можно удалить или оставить 0

    true_direction = 1 if row["CLOSE"] > row["OPEN"] else 0
    predicted_direction = row["PREDICTION"]

    difference = abs(row["CLOSE"] - row["OPEN"])
    return difference if true_direction == predicted_direction else -difference


# Установка рабочей директории в папку, где находится файл скрипта
script_dir = Path(__file__).parent
os.chdir(script_dir)

db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day_2014.db')

for counter in range(1, 101):
    # === ЗАГРУЗКА ДАННЫХ ДЛЯ ВАЛИДАЦИОННОГО ГРАФИКА ===-------------------------------------------
    start_date = '2014-01-01'
    end_date = '2024-01-01'
    df_fut = data_load(db_path, start_date, end_date)

    # === СОЗДАНИЕ ДАТА ФРЕЙМА С ФИЧАМИ И ТАРГЕТОМ ===
    feature_candle_columns = [col for col in df_fut.columns if col.startswith('CI_')]
    feature_volume_columns = [col for col in df_fut.columns if col.startswith('VOL_')]
    feature_day_columns = [col for col in df_fut.columns if col.startswith('DAY_W_')]
    feature_dd_columns = [col for col in df_fut.columns if col.startswith('DD_')]
    feature_io_columns = [col for col in df_fut.columns if col.startswith('IO_')]

    X_candle = df_fut[feature_candle_columns].values.astype(np.int64)
    X_volume = df_fut[feature_volume_columns].values.astype(np.float32)
    X_day = df_fut[feature_day_columns].values.astype(np.int64)
    X_dd = df_fut[feature_dd_columns].values.astype(np.int64)
    X_io = df_fut[feature_io_columns].values.astype(np.float32)

    # === ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(fr"model\best_model_{counter}.pth")
    model = CandleLSTM(
        vocab_size=27, embedding_dim=8, 
        day_vocab_size=7, day_embedding_dim=4, 
        dd_vocab_size=104, dd_embedding_dim=4,   # Увеличено до максимального значения X_dd + 1
        hidden_dim=32, output_dim=1
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === ПОДГОТОВКА ДАННЫХ ДЛЯ ПРЕДСКАЗАНИЯ ===
    X_candle_tensor = torch.tensor(X_candle, dtype=torch.long).to(device)
    X_volume_tensor = torch.tensor(X_volume, dtype=torch.float32).to(device)
    X_day_tensor = torch.tensor(X_day, dtype=torch.long).to(device)
    X_dd_tensor = torch.tensor(X_dd, dtype=torch.long).to(device)
    X_io_tensor = torch.tensor(X_io, dtype=torch.float32).to(device)

    # === ПРОГНОЗ ===
    with torch.no_grad():
        predictions = model(X_candle_tensor, X_volume_tensor, X_day_tensor, X_dd_tensor, X_io_tensor).cpu().numpy()

    df_fut['PREDICTION'] = (predictions > 0.5).astype(int)

    # === РАЗДЕЛЕНИЕ НА ВАЛИДАЦИЮ ===
    split = int(len(df_fut) * 0.85)
    df_val = df_fut.iloc[split:].copy()
    df_val["RESULT"] = df_val.apply(calculate_result, axis=1)
    df_val["CUMULATIVE_RESULT"] = df_val["RESULT"].cumsum()

    # === ЗАГРУЗКА ДАННЫХ ДЛЯ ТЕСТОВАГО ГРАФИКА ===------------------------------------------------
    start_date = '2023-01-01'
    end_date = '2025-03-10'
    df_fut = data_load(db_path, start_date, end_date)

    # === СОЗДАНИЕ ДАТА ФРЕЙМА С ФИЧАМИ И ТАРГЕТОМ ===
    feature_candle_columns = [col for col in df_fut.columns if col.startswith('CI_')]
    feature_volume_columns = [col for col in df_fut.columns if col.startswith('VOL_')]
    feature_day_columns = [col for col in df_fut.columns if col.startswith('DAY_W_')]
    feature_dd_columns = [col for col in df_fut.columns if col.startswith('DD_')]
    feature_io_columns = [col for col in df_fut.columns if col.startswith('IO_')]

    X_candle = df_fut[feature_candle_columns].values.astype(np.int64)
    X_volume = df_fut[feature_volume_columns].values.astype(np.float32)
    X_day = df_fut[feature_day_columns].values.astype(np.int64)
    X_dd = df_fut[feature_dd_columns].values.astype(np.int64)
    X_io = df_fut[feature_io_columns].values.astype(np.float32)

    # === ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(fr"model\best_model_{counter}.pth")
    model = CandleLSTM(
        vocab_size=27, embedding_dim=8, 
        day_vocab_size=7, day_embedding_dim=4, 
        dd_vocab_size=104, dd_embedding_dim=4,   # Увеличено до максимального значения X_dd + 1
        hidden_dim=32, output_dim=1
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === ПОДГОТОВКА ДАННЫХ ДЛЯ ПРЕДСКАЗАНИЯ ===
    X_candle_tensor = torch.tensor(X_candle, dtype=torch.long).to(device)
    X_volume_tensor = torch.tensor(X_volume, dtype=torch.float32).to(device)
    X_day_tensor = torch.tensor(X_day, dtype=torch.long).to(device)
    X_dd_tensor = torch.tensor(X_dd, dtype=torch.long).to(device)
    X_io_tensor = torch.tensor(X_io, dtype=torch.float32).to(device)

    # === ПРОГНОЗ ===
    with torch.no_grad():
        predictions = model(X_candle_tensor, X_volume_tensor, X_day_tensor, X_dd_tensor, X_io_tensor).cpu().numpy()

    df_fut['PREDICTION'] = (predictions > 0.5).astype(int)

    # === РАЗДЕЛЕНИЕ НА ТЕСТ где TRADEDATE больше 2024-01-01 ===
    df_test = df_fut[df_fut['TRADEDATE'] > '2024-01-01'].copy()
    df_test["RESULT"] = df_test.apply(calculate_result, axis=1)
    df_test["CUMULATIVE_RESULT"] = df_test["RESULT"].cumsum()

    # === СОХРАНЕНИЕ ГРАФИКОВ === -----------------------------------------------------------------
    # === ПОСТРОЕНИЕ КУМУЛЯТИВНОГО ГРАФИКА ===
    # Создание фигуры
    plt.figure(figsize=(14, 12))

    # Первый подграфик
    plt.subplot(2, 1, 1)  # (количество строк, количество столбцов, индекс графика)
    plt.plot(df_val["TRADEDATE"], df_val[f"CUMULATIVE_RESULT"], label="Cumulative Result", 
             color="b")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Result")
    plt.title(f"Валидация Sum RTS. set_seed={counter}")
    plt.legend()
    plt.grid()
    plt.xticks(df_val["TRADEDATE"][::15], rotation=90)

    # Второй подграфик
    plt.subplot(2, 1, 2)  # (количество строк, количество столбцов, индекс графика)
    plt.plot(df_test["TRADEDATE"], df_test[f"CUMULATIVE_RESULT"], label="Cumulative Result", 
             color="b")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Result")
    plt.title(f"Независимый тест Sum RTS. set_seed={counter}")
    plt.legend()
    plt.grid()
    plt.xticks(df_test["TRADEDATE"][::10], rotation=90)

    # Сохранение графика в файл
    plt.tight_layout()
    img_path = Path(fr"chart_2/s_{counter}_RTS.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"✅ График сохранен в файл: '{img_path}'. Шаг: {counter}")
    # plt.show()
    plt.close()  # Закрываем текущую фигуру
    