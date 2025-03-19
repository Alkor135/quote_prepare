import pandas as pd
from sklearn.utils import resample
from pathlib import Path
import numpy as np
import sqlite3
import os


def code_int():
    """ Создание словаря кодов свечей """
    combination_dict = {}  # Создаем пустой словарь

    # Генерируем все комбинации от 000 до 222
    for i in range(3):  # Первое число
        for j in range(3):  # Второе число
            for k in range(3):  # Третье число
                # Формируем ключ в виде строки 'ijk'
                key = f"{i}{j}{k}"
                # Значение — порядковый номер комбинации
                value = i * 9 + j * 3 + k  # Вычисляем индекс комбинации
                # Добавляем в словарь
                combination_dict[key] = value
    return combination_dict


def encode_candle(row):
    """ ФУНКЦИЯ КОДИРОВАНИЯ СВЕЧЕЙ (ЛИХОВИДОВ) """
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


def normalize(series):
    """ Функция нормализации от 0 до 1 """
    if len(series) == 0:  # если нет данных, возвращаем NaN
        return None
    return (series.iloc[-1] - series.min()) / (series.max() - series.min()) if series.max() != series.min() else 0


def data_load(db_path, start_date, end_date):
    # SQL-запрос с параметризацией
    query = """
        SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME, LSTTRADE, OPENPOSITION 
        FROM Futures 
        WHERE TRADEDATE BETWEEN ? AND ? 
        ORDER BY TRADEDATE
    """
    # Выполнение запроса
    with sqlite3.connect(db_path) as conn:
        df_fut = pd.read_sql_query(query, conn, params=(start_date, end_date))

    # === 📌 1. СОЗДАНИЕ ПРИЗНАКОВ ИЗ CANDLE CODE ===
    df_fut['CANDLE_CODE'] = df_fut.apply(encode_candle, axis=1)  # Создание кодов свечей по Лиховидову
    # Преобразуем свечные коды в числовой формат (список уникальных кодов)
    code_to_int_dic = code_int()
    df_fut['CANDLE_INT'] = df_fut['CANDLE_CODE'].map(code_to_int_dic)
    # Создание колонок с признаками 'CANDLE_INT' за 20 предыдущих свечей
    for i in range(1, 21):
        df_fut[f'CI_{i}'] = df_fut['CANDLE_INT'].shift(i).astype('Int64')
    # Удаление колонок CANDLE_CODE и CANDLE_INT
    df_fut = df_fut.drop(columns=['CANDLE_CODE', 'CANDLE_INT'])

    # === 📌 2. СОЗДАНИЕ ПРИЗНАКОВ ИЗ VOLUME ===
    # Создаем колонку VOL-NORM
    df_fut['VOL-NORM'] = df_fut['VOLUME'].shift(1).rolling(window=20, min_periods=1).apply(normalize, raw=False)
    # Создание колонок с объемом за 20 предыдущих свечей
    for i in range(1, 21):
        df_fut[f'VOL_{i}'] = df_fut['VOL-NORM'].shift(i)  # .astype('Int64')
    # Удаление колонок VOL-NORM и VOLUME
    df_fut = df_fut.drop(columns=['VOL-NORM', 'VOLUME'])

    # === 📌 3. СОЗДАНИЕ ПРИЗНАКОВ ИЗ TRADEDATE ДНЕЙ НЕДЕЛИ ===
    # Преобразование колонки TRADEDATE в формат datetime
    df_fut['TRADEDATE'] = pd.to_datetime(df_fut['TRADEDATE'])
    # Создание новой колонки DAY_W с номером дня недели
    df_fut['DAY-W'] = df_fut['TRADEDATE'].dt.weekday
    # Создание колонок с днями за 20 предыдущих свечей
    for i in range(1, 21):
        df_fut[f'DAY_W_{i}'] = df_fut['DAY-W'].shift(i).astype('Int64')
    # Удаление колонок DAY-W
    df_fut = df_fut.drop(columns=['DAY-W'])

    # === 📌 4. СОЗДАНИЕ ПРИЗНАКОВ ИЗ LSTTRADE ДНЕЙ ДО ЭКСПИРАЦИИ ===
    # Преобразуем в datetime
    df_fut['TRADEDATE'] = pd.to_datetime(df_fut['TRADEDATE'])
    df_fut['LSTTRADE'] = pd.to_datetime(df_fut['LSTTRADE'])
    # Вычисляем разницу в днях
    df_fut['DATE-DIFF'] = (df_fut['LSTTRADE'] - df_fut['TRADEDATE']).dt.days
    for i in range(1, 21):
        df_fut[f'DD_{i}'] = df_fut['DATE-DIFF'].shift(i).astype('Int64')
    # Удаление колонок DATE-DIFF и LSTTRADE
    df_fut = df_fut.drop(columns=['DATE-DIFF', 'LSTTRADE'])

    # === 📌 5. СОЗДАНИЕ ПРИЗНАКОВ ИЗ OPENPOSITION ===
    # Создаем колонку IO-NORM
    df_fut['IO-NORM'] = df_fut['OPENPOSITION'].shift(1).rolling(window=20, min_periods=1).apply(normalize, raw=False)
    # Создание колонок с объемом за 20 предыдущих свечей
    for i in range(1, 21):
        df_fut[f'IO_{i}'] = df_fut['IO-NORM'].shift(i)  # .astype('Int64')
    # Удаление колонок IO-NORM и OPENPOSITION
    df_fut = df_fut.drop(columns=['IO-NORM', 'OPENPOSITION'])

    # 📌 Создание колонки направления.
    df_fut['DIRECTION'] = (df_fut['CLOSE'] > df_fut['OPEN']).astype(int)

    df_fut = df_fut.dropna().reset_index(drop=True).copy()

    # Принудительное приведение типов (если есть проблемы)
    for col in [f'CI_{i}' for i in range(1, 21)]:
        df_fut[col] = pd.to_numeric(df_fut[col], errors='coerce').fillna(0).astype(np.int64)

    for col in [f'VOL_{i}' for i in range(1, 21)]:
        df_fut[col] = pd.to_numeric(df_fut[col], errors='coerce').fillna(0).astype(np.float32)

    for col in [f'DAY_W_{i}' for i in range(1, 21)]:
        df_fut[col] = pd.to_numeric(df_fut[col], errors='coerce').fillna(0).astype(np.int64)

    for col in [f'DD_{i}' for i in range(1, 21)]:
        df_fut[col] = pd.to_numeric(df_fut[col], errors='coerce').fillna(0).astype(np.int64)

    for col in [f'IO_{i}' for i in range(1, 21)]:
        df_fut[col] = pd.to_numeric(df_fut[col], errors='coerce').fillna(0).astype(np.float32)

    df_fut['DIRECTION'] = pd.to_numeric(df_fut['DIRECTION'], errors='coerce').fillna(0).astype(np.int64)


    return df_fut


if __name__ == '__main__':
    # Установка рабочей директории в папку, где находится файл скрипта
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day_2014.db')

    # Переменные с датами
    start_date = '2014-01-01'
    end_date = '2024-01-01'

    df_fut = data_load(db_path, start_date, end_date)
    print(df_fut)
