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
import matplotlib.pyplot as plt
import os
import json

# === ФУНКЦИЯ КОДИРОВАНИЯ СВЕЧЕЙ (ЛИХОВИДОВ) ===
def encode_candle(row):
    open_, low, high, close = row['OPEN'], row['LOW'], row['HIGH'], row['CLOSE']

    direction = 1 if close > open_ else (0 if close < open_ else 2)
    upper_shadow = high - max(open_, close)
    lower_shadow = min(open_, close) - low
    body = abs(close - open_)

    def classify_shadow(shadow, body):
        return 0 if shadow < 0.1 * body else (1 if shadow < 0.5 * body else 2)

    return (f"{direction}{classify_shadow(upper_shadow, body)}"
            f"{classify_shadow(lower_shadow, body)}")


# Установка рабочей директории в папку, где находится файл скрипта
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Определение путей
db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_day_full.db')

# === ЗАГРУЗКА ДАННЫХ ===
with sqlite3.connect(db_path) as conn:
    df_fut = pd.read_sql_query(
        """
        SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME 
        FROM Day 
        ORDER BY TRADEDATE
        """,
        conn
    )

print(f'Всего записей: {len(df_fut)}')
print(
    f'Дата начала выборки: {df_fut['TRADEDATE'].iloc[0] if len(df_fut) > 0 else None}\n'
    f'Дата окончания выборки: {df_fut['TRADEDATE'].iloc[-1] if len(df_fut) > 0 else None}'
    )

# 1. Создание кодов свечей
df_fut['CANDLE_CODE'] = df_fut.apply(encode_candle, axis=1)

# 2. Формирование единого словаря кодов
unique_codes = sorted(df_fut['CANDLE_CODE'].unique())  # Все уникальные коды свечей
code_to_int = {code: i for i, code in enumerate(unique_codes)}  # Глобальный словарь
# Теперь code_to_int содержит фиксированные числовые метки для каждого свечного кода.
df_fut['CANDLE_INT'] = df_fut['CANDLE_CODE'].map(code_to_int)

# Список уникальных значений
unique_values = df_fut['CANDLE_INT'].unique()
print("Уникальные значения:", unique_values)

# Количество уникальных значений
unique_count = df_fut['CANDLE_INT'].nunique()
print("Количество уникальных значений:", unique_count)

# Частота встречаемости уникальных значений
value_counts = df_fut['CANDLE_INT'].value_counts()
print("Частота встречаемости уникальных значений:\n", value_counts)

# Частота встречаемости (с учетом NaN)
value_counts_with_nan = df_fut['CANDLE_INT'].value_counts(dropna=False)
print("Частота встречаемости уникальных значений (с NaN):\n", value_counts_with_nan)

# 3. Сохранение словаря в JSON
with open("code_to_int.json", "w") as f:
    json.dump(code_to_int, f)

"""
Использование глобального сохраненного словаря.
import json
# Загрузка словаря
with open("code_to_int.json", "r") as f:
    code_to_int = json.load(f)

# Кодируем новые данные, используя старый словарь
df_new['CANDLE_INT'] = df_new['CANDLE_CODE'].map(code_to_int)
"""

print(df_fut)
