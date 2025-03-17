"""
Сохранение графика полной выборки в файл.
"""
import sqlite3
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import os
from sklearn.preprocessing import StandardScaler
# Импортируем кодировку свечей
from data_processing import data_prepare


# === ОПРЕДЕЛЕНИЕ МОДЕЛИ LSTM (ДОЛЖНА СОВПАДАТЬ С ОБУЧЕННОЙ) ===
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
        # x = torch.cat((x_candle, x_volume[..., None], x_day), dim=-1)
        # x = torch.cat((x_candle, x_volume.view(x_volume.shape[0], x_volume.shape[1], 1), x_day), dim=-1)
        # print(f"x_volume.shape после unsqueeze: {x_volume.unsqueeze(-1).shape}")

        # Пропускаем через LSTM
        x, _ = self.lstm(x)

        # Полносвязный слой и сигмоида
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

db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_day_full.db')

for counter in range(1, 101):
    # === ЗАГРУЗКА ДАННЫХ ДЛЯ ВАЛИДАЦИОННОГО ГРАФИКА ===-------------------------------------------
    with sqlite3.connect(db_path) as conn:
        df_fut = pd.read_sql_query(
            """
            SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME 
            FROM Day 
            WHERE TRADEDATE >= '2014-01-01' 
            ORDER BY TRADEDATE
            """,
            conn
        )

    df_fut = data_prepare(df_fut)

    # === СОЗДАНИЕ ДАТА ФРЕЙМА С ФИЧАМИ И ТАРГЕТОМ ===
    feature_candle_columns = [col for col in df_fut.columns if col.startswith('CI_')]
    feature_volume_columns = [col for col in df_fut.columns if col.startswith('VOL_')]
    feature_day_columns = [col for col in df_fut.columns if col.startswith('DAY_W_')]

    X_candle = df_fut[feature_candle_columns].values.astype(np.int64)
    X_volume = df_fut[feature_volume_columns].values.astype(np.int64)
    X_day = df_fut[feature_day_columns].values.astype(np.int64)

    # === ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(fr"model\best_model_{counter}.pth")
    model = CandleLSTM(vocab_size=27, embedding_dim=8, day_vocab_size=7, day_embedding_dim=4, hidden_dim=32, output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === ПОДГОТОВКА ДАННЫХ ДЛЯ ПРЕДСКАЗАНИЯ ===
    X_candle_tensor = torch.tensor(X_candle, dtype=torch.long).to(device)
    X_volume_tensor = torch.tensor(X_volume, dtype=torch.float32).to(device)
    X_day_tensor = torch.tensor(X_day, dtype=torch.long).to(device)

    # === ПРОГНОЗ ===
    with torch.no_grad():
        predictions = model(X_candle_tensor, X_volume_tensor, X_day_tensor).cpu().numpy()

    df_fut['PREDICTION'] = (predictions > 0.5).astype(int)

    df_fut = df_fut.copy()
    df_fut["RESULT"] = df_fut.apply(calculate_result, axis=1)
    df_fut["CUMULATIVE_RESULT"] = df_fut["RESULT"].cumsum()

    # === СОХРАНЕНИЕ ГРАФИКОВ === -----------------------------------------------------------------
    # === ПОСТРОЕНИЕ КУМУЛЯТИВНОГО ГРАФИКА ===
    # Создание фигуры
    df_fut["CUMULATIVE_RESULT"] = df_fut["RESULT"].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(df_fut["TRADEDATE"], df_fut["CUMULATIVE_RESULT"], label="Cumulative Result",
             color="b")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Result")
    plt.title(f"Cumulative Sum RTS. set_seed={counter}")
    plt.legend()
    plt.grid()

    plt.xticks(df_fut["TRADEDATE"][::100], rotation=90)
    # Сохранение графика в файл
    img_path = Path(fr"chart_full\s_{counter}_RTS.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"✅ График сохранен в файл: '{img_path}'")
    # plt.show()
    plt.close()  # Закрываем текущую фигуру
