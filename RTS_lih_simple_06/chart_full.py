"""
Сохранение 2 графиков (валидационного и тестового) в файл.
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
from data_processing import encode_candle


# === ОПРЕДЕЛЕНИЕ МОДЕЛИ LSTM (ДОЛЖНА СОВПАДАТЬ С ОБУЧЕННОЙ) ===
class CandleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(CandleLSTM, self).__init__()

        # Embedding слой для кодов свечей
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM принимает объединенные фичи (embedding + volume)
        self.lstm = nn.LSTM(embedding_dim + 1, hidden_dim, batch_first=True)

        # Полносвязный слой для предсказания
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_candle, x_volume):
        # Преобразуем коды свечей в embedding
        x_candle = self.embedding(x_candle)

        # Объединяем свечи и объем (по оси признаков)
        x = torch.cat((x_candle, x_volume.unsqueeze(-1)), dim=-1)

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

# Загрузка полного словаря
with open("code_full_int.json", "r") as f:
    code_to_int = json.load(f)

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

    # Создание кодов свечей по Лиховидову
    df_fut['CANDLE_CODE'] = df_fut.apply(encode_candle, axis=1)

    # === 4. ПОДГОТОВКА ДАННЫХ ===
    # Преобразуем свечные коды в числовой формат (список уникальных кодов)
    df_fut['CANDLE_INT'] = df_fut['CANDLE_CODE'].map(code_to_int)

    # Создание колонки направления.
    df_fut['DIRECTION'] = (df_fut['CLOSE'] > df_fut['OPEN']).astype(int)

    # Создание колонок с признаками
    for i in range(1, 21):
        df_fut[f'CI_{i}'] = df_fut['CANDLE_INT'].shift(i).astype('Int64')
    
    # Создание колонок с объемом за 20 предыдущих свечей
    for i in range(1, 21):
        df_fut[f'VOL_{i}'] = df_fut['VOLUME'].shift(i).astype('Int64')

    df_fut = df_fut.dropna().reset_index(drop=True)

    # === СОЗДАНИЕ ДАТА ФРЕЙМА С ФИЧАМИ И ТАРГЕТОМ ===
    feature_candle_columns = [col for col in df_fut.columns if col.startswith('CI_')]
    feature_volume_columns = [col for col in df_fut.columns if col.startswith('VOL_')]

    X_candle = df_fut[feature_candle_columns].values.astype(np.int64)
    X_volume = df_fut[feature_volume_columns].values.astype(np.float32)

    # === НОРМАЛИЗАЦИЯ ОБЪЕМОВ ===
    scaler = StandardScaler()
    X_volume = np.array([scaler.fit_transform(row.reshape(-1, 1)).flatten() for row in X_volume])

    # === ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(fr"model\best_model_{counter}.pth")
    model = CandleLSTM(vocab_size=27, embedding_dim=8, hidden_dim=32, output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === ПОДГОТОВКА ДАННЫХ ДЛЯ ПРЕДСКАЗАНИЯ ===
    X_candle_tensor = torch.tensor(X_candle, dtype=torch.long).to(device)
    X_volume_tensor = torch.tensor(X_volume, dtype=torch.float32).to(device)

    # === ПРОГНОЗ ===
    with torch.no_grad():
        predictions = model(X_candle_tensor, X_volume_tensor).cpu().numpy()

    df_fut['PREDICTION'] = (predictions > 0.5).astype(int)

    # === РАЗДЕЛЕНИЕ НА ВАЛИДАЦИЮ ===
    # split = int(len(df_fut) * 0.85)
    # df_val = df_fut.iloc[split:].copy()
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
    