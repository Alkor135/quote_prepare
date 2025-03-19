"""
Для обучения моделей с разными гиперпараметрами. Лиховидов. Бинарка.
С балансировкой классов добавлением рандомных, где нет совпадения по фичам 
с противоположным классом.
Лучшая модель сохраняется по Profit - Loss критерию.
Только тест.
Сохранение дата фрейма в файл для анализа.
"""
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
import json
import os


# === ФУНКЦИЯ КОДИРОВАНИЯ СВЕЧЕЙ (ЛИХОВИДОВ) ===
def encode_candle(row):
    open_, low, high, close = row['OPEN'], row['LOW'], row['HIGH'], row['CLOSE']

    if close > open_:
        direction = 1  # Бычья свеча
    elif close < open_:
        direction = 0  # Медвежья свеча
    else:
        direction = 2  # Дожи

    upper_shadow = high - max(open_, close)
    lower_shadow = min(open_, close) - low
    body = abs(close - open_)

    def classify_shadow(shadow, body):
        if shadow < 0.1 * body:
            return 0  
        elif shadow < 0.5 * body:
            return 1  
        else:
            return 2  

    upper_code = classify_shadow(upper_shadow, body)
    lower_code = classify_shadow(lower_shadow, body)

    return f"{direction}{upper_code}{lower_code}"


# === СОЗДАНИЕ НЕЙРОСЕТИ (LSTM) (ДОЛЖНА СОВПАДАТЬ С ОБУЧЕННОЙ) ===
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
data_path = Path(fr"pred_res_cum.csv")
df_data = pd.DataFrame()

for counter in range(1, 101):
    # ---------------------------------------------------------------------------------------------
    # === 1. ЗАГРУЗКА ДАННЫХ ===
    with sqlite3.connect(db_path) as conn:
        df_fut = pd.read_sql_query(
            """
            SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME 
            FROM Day 
            WHERE TRADEDATE >= '2023-01-01' 
            ORDER BY TRADEDATE
            """,
            conn
        )

    df_fut['CANDLE_CODE'] = df_fut.apply(encode_candle, axis=1)

    # === 3. ПРЕОБРАЗОВАНИЕ КОДОВ В ЧИСЛА ===
    df_fut['CANDLE_INT'] = df_fut['CANDLE_CODE'].map(code_to_int)

    window_size = 20

    # === 5. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(fr"model\best_model_{counter}.pth")  #
    model = CandleLSTM(vocab_size=27, embedding_dim=8, hidden_dim=32, output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 6. ПРОГНОЗИРОВАНИЕ ===
    predictions = []
    with torch.no_grad():
        for i in range(len(df_fut) - window_size):
            sequence = torch.tensor(
                df_fut['CANDLE_INT'].iloc[i:i + window_size].values, dtype=torch.long
            ).unsqueeze(0).to(device)
            pred = model(sequence).item()
            predictions.append(1 if pred > 0.5 else 0)

    # Заполняем колонку PREDICTION (первые window_size значений - NaN)
    df_fut[f'PRED_{counter}'] = [None] * window_size + predictions

    # Выбор строк, где TRADEDATE больше 2024-01-01
    df = df_fut[df_fut['TRADEDATE'] > '2024-01-01'].copy()

    # === 3. РАСЧЁТ РЕЗУЛЬТАТОВ ПРОГНОЗА ===
    def calculate_result(row):
        if pd.isna(row[f"PRED_{counter}"]):  # Если NaN после сдвига
            return 0  # Можно удалить или оставить 0

        true_direction = 1 if row["CLOSE"] > row["OPEN"] else 0
        predicted_direction = row[f"PRED_{counter}"]

        difference = abs(row["CLOSE"] - row["OPEN"])
        return difference if true_direction == predicted_direction else -difference


    df[f"RES_{counter}"] = df.apply(calculate_result, axis=1)
    df[f"CUM_{counter}"] = df[f"RES_{counter}"].cumsum()
    # print('Проверка counter')
    # print(df)

    if counter == 1:
        df_data = df
        # print('Копирование')
    else:
        # Выполняем правое слияние
        # print('Слияние')
        df_data = pd.merge(
            df_data,
            df,
            how='right',  # правое слияние
            on=['TRADEDATE', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLUME', 'CANDLE_CODE',
                'CANDLE_INT'],
            suffixes=('_x', '_y')  # добавляем суффиксы для различения одинаковых колонок
        )

    # === 4. Сохранение данных в файл ===
    df_data.to_csv(data_path, index=False)
    print(f"✅ Данные сохранены в файл: '{data_path}', {counter=}\n")
    # print(df_data)
