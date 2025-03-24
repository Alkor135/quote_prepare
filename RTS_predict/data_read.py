"""
Загрузка данных по фьючерсам из базы данных SQLite.
Создание признаков на основе кодов свечей Лиховидова.
"""

import pandas as pd
# from sklearn.utils import resample
from pathlib import Path
import numpy as np
import sqlite3
import os
import sys
sys.dont_write_bytecode = True


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


def data_load(db_path, start_date):
    """
    Загрузка данных из базы данных, начиная с указанной даты.
    """
    query = """
        SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE
        FROM Futures
        WHERE TRADEDATE >= ?
        ORDER BY TRADEDATE ASC
    """
    try:
        with sqlite3.connect(db_path) as conn:
            df_fut = pd.read_sql_query(query, conn, params=(start_date,))
    except sqlite3.Error as e:
        raise RuntimeError(f"Ошибка подключения к базе данных: {e}")

    # Преобразуем TRADEDATE в datetime
    df_fut['TRADEDATE'] = pd.to_datetime(df_fut['TRADEDATE'])

    # Сортируем записи в хронологическом порядке
    df_fut = df_fut.sort_values(by='TRADEDATE').reset_index(drop=True)

    # === 📌 1. СОЗДАНИЕ ПРИЗНАКОВ ИЗ CANDLE CODE ===
    df_fut['CANDLE_CODE'] = df_fut.apply(encode_candle, axis=1)  # Создание кодов свечей по Лиховидову
    
    # Преобразуем свечные коды в числовой формат (список уникальных кодов)
    code_to_int_dic = code_int()
    df_fut['CANDLE_INT'] = df_fut['CANDLE_CODE'].map(code_to_int_dic)

    # Создание колонок с признаками 'CANDLE_INT' за 20 предыдущих свечей
    shifts = {f'CI_{i}': df_fut['CANDLE_INT'].shift(i) for i in range(1, 21)}
    df_fut = pd.concat([df_fut, pd.DataFrame(shifts)], axis=1)
    # for i in range(1, 21):
    #     df_fut[f'CI_{i}'] = df_fut['CANDLE_INT'].shift(i).astype('Int64')

    # Удаление колонок CANDLE_CODE и CANDLE_INT
    df_fut = df_fut.drop(columns=['CANDLE_CODE', 'CANDLE_INT'])

    # 📌 Создание колонки направления.
    df_fut['DIRECTION'] = (df_fut['CLOSE'] > df_fut['OPEN']).astype(int)

    df_fut = df_fut.dropna().reset_index(drop=True).copy()

    # Принудительное приведение типов (если есть проблемы)
    for col in [f'CI_{i}' for i in range(1, 21)]:
        df_fut[col] = pd.to_numeric(df_fut[col], errors='coerce').fillna(0).astype(np.int64)

    df_fut['DIRECTION'] = pd.to_numeric(
        df_fut['DIRECTION'], errors='coerce'
        ).fillna(0).astype(np.int64)

    df_fut = df_fut.copy()

    return df_fut


if __name__ == '__main__':
    # Установка рабочей директории в папку, где находится файл скрипта
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day_2014.db')
    start_date = '2023-01-01'  # Начальная дата

    df_fut = data_load(db_path, start_date)
    print(df_fut)
