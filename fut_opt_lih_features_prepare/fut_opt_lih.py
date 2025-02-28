import sqlite3
from pathlib import Path
import numpy as np

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Настройки для отображения широкого df pandas
pd.options.display.width = 1200
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 100

# Загрузка данных -------------------------------------------------------------
# Путь к базе данных SQLite
db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day.db')
# Загрузка данных
with sqlite3.connect(db_path) as conn:
    df_fut = pd.read_sql_query(
        "SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME, OPENPOSITION, LSTTRADE FROM Futures", 
        conn
    )
    df_opt = pd.read_sql_query(
        "SELECT TRADEDATE, OPENPOSITION, OPTIONTYPE, STRIKE FROM Options", 
        conn
    )

# Функция кодирования свечи по Лиховидову
def encode_candle(row):
    open_, low, high, close = row['OPEN'], row['LOW'], row['HIGH'], row['CLOSE']
    
    # Определяем направление свечи
    if close > open_:
        direction = 1  # Бычья свеча
    elif close < open_:
        direction = 0  # Медвежья свеча
    else:
        direction = 2  # Дожи

    # Определяем длину теней
    upper_shadow = high - max(open_, close)
    lower_shadow = min(open_, close) - low
    body = abs(close - open_)  # Длина тела свечи

    # Функция классификации тени
    def classify_shadow(shadow, body):
        if shadow < 0.1 * body:
            return 0  # Короткая тень
        elif shadow < 0.5 * body:
            return 1  # Средняя тень
        else:
            return 2  # Длинная тень

    upper_code = classify_shadow(upper_shadow, body)
    lower_code = classify_shadow(lower_shadow, body)

    # Формируем итоговый код свечи
    return f"{direction}{upper_code}{lower_code}"

# Добавляем колонку с кодом свечи
df_fut['CANDLE_CODE'] = df_fut.apply(encode_candle, axis=1)

# Вывод первых 10 строк для проверки
print(df_fut[['TRADEDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'CANDLE_CODE']].head(10))

