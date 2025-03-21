import pandas as pd
from pathlib import Path
import numpy as np
import sqlite3
import os
import sys
sys.dont_write_bytecode = True

def normalize(series):
    """ Функция нормализации от 0 до 1 """
    if len(series) == 0:  # если нет данных, возвращаем NaN
        return None
    return (series.iloc[-1] - series.min()) / (series.max() - series.min()) if series.max() != series.min() else 0


def data_load(db_path, start_date, end_date):
    # Чтение данных по фьючерсам
    query = """
        SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE 
        FROM Futures 
        WHERE TRADEDATE BETWEEN ? AND ? 
        ORDER BY TRADEDATE
    """
    with sqlite3.connect(db_path) as conn:
        df_fut = pd.read_sql_query(query, conn, params=(start_date, end_date))

    # Чтение данных по фьючерсам
    query = """
        SELECT TRADEDATE, OPENPOSITION, OPTIONTYPE, STRIKE, LSTTRADE 
        FROM Options
        WHERE TRADEDATE BETWEEN ? AND ? 
        ORDER BY TRADEDATE
    """
    with sqlite3.connect(db_path) as conn:
        df_opt = pd.read_sql_query(query, conn, params=(start_date, end_date))

    # df_fut['DIRECTION'] = (df_fut['CLOSE'] > df_fut['OPEN']).astype(int)

    df_fut[['TRADEDATE']] = df_fut[['TRADEDATE']].apply(pd.to_datetime)
    df_opt[['TRADEDATE', 'LSTTRADE']] = df_opt[['TRADEDATE', 'LSTTRADE']].apply(pd.to_datetime)

    df_opt = df_opt[df_opt.TRADEDATE < df_opt.LSTTRADE]

    step_strike = 2500
    df_rez = pd.DataFrame()

    for row in df_fut.itertuples(index=False):  
        df = df_opt[df_opt.TRADEDATE == row.TRADEDATE]

        df_p = (df[df.OPTIONTYPE == "P"]
                .groupby('STRIKE', as_index=False)['OPENPOSITION']
                .sum()
                .rename(columns={'OPENPOSITION': 'oi_p'}))

        df_c = (df[df.OPTIONTYPE == "C"]
                .groupby('STRIKE', as_index=False)['OPENPOSITION']
                .sum()
                .rename(columns={'OPENPOSITION': 'oi_c'}))

        if df_p.empty and df_c.empty:
            continue

        # Создание DataFrame со страйками
        df_tmp = pd.DataFrame({'STRIKE': np.arange(df.STRIKE.min(), df.STRIKE.max() + step_strike, step_strike)})

        # Объединение данных
        merged_df = df_tmp.merge(df_p, on='STRIKE', how='left').merge(df_c, on='STRIKE', how='left').fillna(0)

        # Приведение типов
        merged_df[['oi_c', 'oi_p']] = merged_df[['oi_c', 'oi_p']].astype(np.int64)

        # Накопленные суммы
        merged_df['oi_c'] = merged_df['oi_c'].cumsum()
        merged_df['oi_p'] = merged_df['oi_p'][::-1].cumsum()[::-1]

        price_close = df_fut.loc[df_fut['TRADEDATE'] == row.TRADEDATE, 'CLOSE'].values[0]
        nearest_strike = round(price_close / step_strike) * step_strike

        if nearest_strike not in merged_df['STRIKE'].values:
            continue  # Если страйк отсутствует, пропускаем

        index_nearest = merged_df.index[merged_df['STRIKE'] == nearest_strike][0]
        start_index = max(0, index_nearest - 3)
        end_index = min(len(merged_df), index_nearest + 4)

        subset_df = merged_df.iloc[start_index:end_index].copy()
        subset_df['oi'] = np.where(subset_df['STRIKE'] < price_close,
                                subset_df['oi_p'] - subset_df['oi_c'],
                                subset_df['oi_c'] - subset_df['oi_p'])

        column_oi_arr = subset_df['oi'].values
        arr_min, arr_max = column_oi_arr.min(), column_oi_arr.max()
        normalized_arr = (column_oi_arr - arr_min) / (arr_max - arr_min) if arr_max != arr_min else np.zeros_like(column_oi_arr)

        row_data = [row.TRADEDATE, row.OPEN, row.LOW, row.HIGH, row.CLOSE, nearest_strike] + list(normalized_arr)
        
        df_rez = pd.concat([df_rez, pd.DataFrame([row_data])], ignore_index=True)

    # Преобразование даты
    df_rez[0] = pd.to_datetime(df_rez[0]).dt.date

    # Переименование колонок
    df_rez = df_rez.rename(columns={
        0: 'TRADEDATE', 1: 'OPEN', 2: 'LOW', 3: 'HIGH', 4: 'CLOSE', 5: 'ZERO_STRIKE', 
        6: 'io_0', 7: 'io_1', 8: 'io_2', 9:'io_3', 10:'io_4', 11:'io_5', 12:'io_6'
        # **{i + 6: v for i, v in enumerate(range(-7500, 7501, 2500))}, 
        # len(df_rez.columns) - 1: 'DIRECTION'
    })

    # Создание колонок с признаками за 10 предыдущих свечей
    for i in range(1, 11):
        df_rez[f'mp0_{i}'] = df_rez['io_0'].shift(i).astype('float32')
    for i in range(1, 11):
        df_rez[f'mp1_{i}'] = df_rez['io_1'].shift(i).astype('float32')
    for i in range(1, 11):
        df_rez[f'mp2_{i}'] = df_rez['io_2'].shift(i).astype('float32')
    for i in range(1, 11):
        df_rez[f'mp3_{i}'] = df_rez['io_3'].shift(i).astype('float32')
    for i in range(1, 11):
        df_rez[f'mp4_{i}'] = df_rez['io_4'].shift(i).astype('float32')
    for i in range(1, 11):
        df_rez[f'mp5_{i}'] = df_rez['io_5'].shift(i).astype('float32')
    for i in range(1, 11):
        df_rez[f'mp6_{i}'] = df_rez['io_6'].shift(i).astype('float32')

    # Удаление колонок CANDLE_CODE и CANDLE_INT
    df_rez = df_rez.drop(columns=['io_0', 'io_1', 'io_2', 'io_3', 'io_4', 'io_5', 'io_6', 'ZERO_STRIKE'])

    # 📌 Создание колонки направления.
    df_rez['DIRECTION'] = (df_rez['CLOSE'] > df_rez['OPEN']).astype(int)

    # Удаляем строки с NaN и сбрасываем индексы
    df_rez.dropna(inplace=True)  # Удаляем строки с NaN
    df_rez.reset_index(inplace=True, drop=True)  # Сбрасываем индексы

    return df_rez


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
