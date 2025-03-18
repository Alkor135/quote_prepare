import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import numpy as np
import sqlite3
import os

def read_data(db_path, start_date, end_date):
    # Чтение данных по фьючерсам
    query = """
        SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME, LSTTRADE, OPENPOSITION 
        FROM Futures 
        WHERE TRADEDATE BETWEEN ? AND ? 
        ORDER BY TRADEDATE
    """
    with sqlite3.connect(db_path) as conn:
        df_fut = pd.read_sql_query(query, conn, params=(start_date, end_date))

    # Чтение данных по опционам
    query = """
        SELECT TRADEDATE, OPENPOSITION, OPTIONTYPE, STRIKE
        FROM Options 
        WHERE TRADEDATE BETWEEN ? AND ? 
        ORDER BY TRADEDATE
    """
    with sqlite3.connect(db_path) as conn:
        df_opt = pd.read_sql_query(query, conn, params=(start_date, end_date))

    # Преобразуем TRADEDATE в datetime
    df_fut['TRADEDATE'] = pd.to_datetime(df_fut['TRADEDATE'])
    df_opt['TRADEDATE'] = pd.to_datetime(df_opt['TRADEDATE'])

    # Объединяем df_opt с df_fut по TRADEDATE, чтобы добавить колонку CLOSE в df_opt
    df_opt = df_opt.merge(df_fut[['TRADEDATE', 'CLOSE']], on='TRADEDATE', how='left')

    # Фильтрация и группировка по условиям
    df_calls_itm = df_opt[(df_opt['OPTIONTYPE'] == 'C') & (df_opt['STRIKE'] < 
        df_opt['CLOSE'])].groupby('TRADEDATE')['OPENPOSITION'].sum().rename('CALLS_ITM')

    df_calls_otm = df_opt[(df_opt['OPTIONTYPE'] == 'C') & (df_opt['STRIKE'] > 
        df_opt['CLOSE'])].groupby('TRADEDATE')['OPENPOSITION'].sum().rename('CALLS_OTM')

    df_puts_itm = df_opt[(df_opt['OPTIONTYPE'] == 'P') & (df_opt['STRIKE'] < 
        df_opt['CLOSE'])].groupby('TRADEDATE')['OPENPOSITION'].sum().rename('PUTS_ITM')

    df_puts_otm = df_opt[(df_opt['OPTIONTYPE'] == 'P') & (df_opt['STRIKE'] > 
        df_opt['CLOSE'])].groupby('TRADEDATE')['OPENPOSITION'].sum().rename('PUTS_OTM')

    # Объединяем агрегированные данные с df_fut
    df = df_fut.merge(df_calls_itm, on='TRADEDATE', how='left')
    df = df.merge(df_calls_otm, on='TRADEDATE', how='left')
    df = df.merge(df_puts_itm, on='TRADEDATE', how='left')
    df = df.merge(df_puts_otm, on='TRADEDATE', how='left')

    # Заполняем NaN нулями (если на дату нет подходящих опционов)
    df.fillna(0, inplace=True)

    # Функция нормализации строки
    def normalize_row(row):
        min_val = row.min()
        max_val = row.max()
        return (row - min_val) / (max_val - min_val) if max_val != min_val else row * 0
    
    # Список колонок, которые нужно нормализовать
    columns_to_normalize = ['CALLS_ITM', 'CALLS_OTM', 'PUTS_ITM', 'PUTS_OTM']
    # Копируем оригинальные столбцы, которые нужно нормализовать
    normalized_data = df[columns_to_normalize].apply(normalize_row, axis=1)
    # Заменяем нормализованные столбцы в оригинальном DataFrame
    df[columns_to_normalize] = normalized_data

    # Вывод результата
    print(df)


if __name__ == '__main__':
    # Установка рабочей директории в папку, где находится файл скрипта
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Переменные с датами
    start_date = '2014-01-01'
    end_date = '2024-01-01'

    db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day_2014.db')

    read_data(db_path, start_date, end_date)
