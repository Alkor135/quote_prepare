import sqlite3
from pathlib import Path
import numpy as np

import pandas as pd
import zipfile


if __name__ == '__main__':
    # Укажите путь к вашей базе данных SQLite
    db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day.db')
    # ---------------------------------------------------------------------------------------------
    # Настройки для отображения широкого df pandas
    pd.options.display.width = 1200
    pd.options.display.max_colwidth = 100
    pd.options.display.max_columns = 100

    # Установите соединение с базой данных
    conn = sqlite3.connect(db_path)

    # Получение df с именами таблиц
    query = "SELECT name FROM sqlite_master WHERE type='table'"
    df_table = pd.read_sql_query(query, conn)
    # table = pd.read_sql_query(query, conn).iloc[0, 0]
    print(df_table)

    # Выполните SQL-запрос и загрузите результаты в DataFrame
    query = f"SELECT * FROM {df_table.iloc[0, 0]}"
    df_fut = pd.read_sql_query(query, conn)
    # Проверьте данные
    print(df_fut.columns.tolist())
    print(df_fut)

    # Выполните SQL-запрос и загрузите результаты в DataFrame
    query = f"SELECT * FROM {df_table.iloc[1, 0]}"
    df_opt = pd.read_sql_query(query, conn)
    print(df_opt.columns.tolist())
    print(df_opt)

    # Закройте соединение
    conn.close()

    # Группируем по дате и типу опциона и суммируем 'OPENPOSITION'
    sums = df_opt.groupby(['TRADEDATE', 'OPTIONTYPE'])['OPENPOSITION'].sum().unstack()
    # sums.rename(columns={'C': 'OI_C', 'P': 'OI_P'}, inplace=True)
    print(sums)

    # Объединяем датафреймы по полю 'TRADEDATE'
    df = pd.merge(df_fut, sums, on='TRADEDATE', how='inner')
    df.rename(columns={'OPENPOSITION': 'OI', 'C': 'OI_C', 'P': 'OI_P'}, inplace=True)
    # Удаляем ненужные колонки, например, 'OPEN' и 'VOLUME'
    df = df.drop(columns=['SECID', 'SHORTNAME', 'LSTTRADE'])
    print(df)

    # Вычисляем скользящее среднее за 10 предыдущих значений, включая текущее
    df['MA_10'] = df['VOLUME'].shift(1).rolling(window=10, min_periods=1).mean()
    # Нормализуем колонку 'VOLUME' к среднему значению от -1 до 1
    df['norm_vol'] = 2 * (df['VOLUME'] - df['MA_10']) / (df['MA_10'] + 1)
    df = df.drop(columns=['MA_10'])

    # # Создаем скользящее окно и вычисляем среднее значение за 10 предыдущих значений (включая текущее)
    # df['rolling_mean'] = df['VOLUME'].rolling(window=10, min_periods=1).mean()
    # # Нормализуем значения относительно среднего
    # df['normalized'] = (df['VOLUME'] - df['rolling_mean']) / df['rolling_mean']
    # # # Нормализуем значения в диапазоне от -1 до 1
    # # df['normalized'] = df['normalized'].clip(-1,1)

    # Вычисляем скользящее среднее за 10 предыдущих значений, включая текущее
    df['MA_10'] = df['OI'].shift(1).rolling(window=10, min_periods=1).mean()
    # Нормализуем колонку к среднему значению от -1 до 1
    df['norm_oi'] = 2 * (df['OI'] - df['MA_10']) / (df['MA_10'] + 1)
    df = df.drop(columns=['MA_10'])

    # Вычисляем скользящее среднее за 10 предыдущих значений, включая текущее
    df['MA_10'] = df['OI_C'].shift(1).rolling(window=10, min_periods=1).mean()
    # Нормализуем колонку к среднему значению от -1 до 1
    df['norm_oi_c'] = 2 * (df['OI_C'] - df['MA_10']) / (df['MA_10'] + 1)
    df = df.drop(columns=['MA_10'])

    # Вычисляем скользящее среднее за 10 предыдущих значений, включая текущее
    df['MA_10'] = df['OI_P'].shift(1).rolling(window=10, min_periods=1).mean()
    # Нормализуем колонку к среднему значению от -1 до 1
    df['norm_oi_p'] = 2 * (df['OI_P'] - df['MA_10']) / (df['MA_10'] + 1)
    df = df.drop(columns=['MA_10'])

    print(df)