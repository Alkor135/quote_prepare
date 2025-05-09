"""
Сохранение графика теста на независимой тестовой выборке в файл.
"""
import sqlite3
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt
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


# === ОПРЕДЕЛЕНИЕ МОДЕЛИ (ДОЛЖНА СОВПАДАТЬ С ОБУЧЕННОЙ) ===
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

db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_day_full.db')

for counter in range(1, 101):
    # === 2. ЗАГРУЗКА ДАННЫХ ===
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
    unique_codes = sorted(df_fut['CANDLE_CODE'].unique())
    code_to_int = {code: i for i, code in enumerate(unique_codes)}
    df_fut['CANDLE_INT'] = df_fut['CANDLE_CODE'].map(code_to_int)

    window_size = 20

    # === 5. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(fr"model\best_model_{counter}.pth")
    model = CandleLSTM(vocab_size=len(unique_codes), embedding_dim=8, hidden_dim=32,
                       output_dim=1).to(device)
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
    df_fut['PREDICTION'] = [None] * window_size + predictions

    # Выбор строк, где TRADEDATE больше 2024-01-01
    df_test = df_fut[df_fut['TRADEDATE'] > '2024-01-01'].copy()

    df_test["RESULT"] = df_test.apply(calculate_result, axis=1)

    # === 4. ПОСТРОЕНИЕ КУМУЛЯТИВНОГО ГРАФИКА ===
    df_test["CUMULATIVE_RESULT"] = df_test["RESULT"].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(df_test["TRADEDATE"], df_test["CUMULATIVE_RESULT"], label="Cumulative Result",
             color="b")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Result")
    plt.title(f"Cumulative Sum RTS. set_seed={counter}")
    plt.legend()
    plt.grid()

    plt.xticks(df_test["TRADEDATE"][::10], rotation=90)
    # Сохранение графика в файл
    img_path = Path(fr"chart_test\s_{counter}_RTS.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"✅ График сохранен в файл: '{img_path}' \n")
    # plt.show()
