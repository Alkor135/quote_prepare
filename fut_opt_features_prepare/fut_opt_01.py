import sqlite3
from pathlib import Path
import numpy as np

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# # Настройки для отображения широкого df pandas
# pd.options.display.width = 1200
# pd.options.display.max_colwidth = 100
# pd.options.display.max_columns = 100

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

# Обработка и слияние данных по фьючерсам и опционам --------------------------
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

# Объединяем агрегированные данные
df = df_fut.merge(df_calls_itm, on='TRADEDATE', how='left')
df = df.merge(df_calls_otm, on='TRADEDATE', how='left')
df = df.merge(df_puts_itm, on='TRADEDATE', how='left')
df = df.merge(df_puts_otm, on='TRADEDATE', how='left')

# Заполняем NaN нулями (если на дату нет подходящих данных)
df.fillna(0, inplace=True)

# Создание фичей --------------------------------------------------------------
# Получение фичей по количеству дней до истечения фьючерса
# Преобразуем в datetime
df['LSTTRADE'] = pd.to_datetime(df['LSTTRADE'])
# Вычисляем разницу в днях
df['date_diff'] = (df['LSTTRADE'] - df['TRADEDATE']).dt.days
# Нормализация в диапазон [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
df['norm_date_diff'] = scaler.fit_transform(df[['date_diff']]).clip(0, 1)
df = df.drop(columns=['date_diff', 'LSTTRADE'])

# Создание фичей из объемов торгов по фьючерсам
# Среднее за 10 предыдущих значений (shift(1), чтобы исключить текущее)
df['VOLUME_MEAN_10'] = df['VOLUME'].shift(1).rolling(window=10, min_periods=1).mean()

# Отношение текущего объема к среднему
df['VOLUME_RATIO'] = df['VOLUME'] / df['VOLUME_MEAN_10']

# Нормализация в диапазон (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
df['norm_vol'] = scaler.fit_transform(df[['VOLUME_RATIO']])

df = df.drop(columns=['VOLUME_MEAN_10', 'VOLUME_RATIO'])

# Новые фичи: Отношение CALLS_ITM, CALLS_OTM, PUTS_ITM, PUTS_OTM к OPENPOSITION
df['CALLS_ITM_RATIO'] = df['CALLS_ITM'] / df['OPENPOSITION']
df['CALLS_OTM_RATIO'] = df['CALLS_OTM'] / df['OPENPOSITION']
df['PUTS_ITM_RATIO'] = df['PUTS_ITM'] / df['OPENPOSITION']
df['PUTS_OTM_RATIO'] = df['PUTS_OTM'] / df['OPENPOSITION']

df[['CALLS_ITM_RATIO', 'CALLS_OTM_RATIO', 'PUTS_ITM_RATIO', 'PUTS_OTM_RATIO']] = df[
    ['CALLS_ITM_RATIO', 'CALLS_OTM_RATIO', 'PUTS_ITM_RATIO', 'PUTS_OTM_RATIO']
    ].fillna(0)

# Нормализация новых фич
df[
    ['CALLS_ITM_RATIO', 'CALLS_OTM_RATIO', 'PUTS_ITM_RATIO', 'PUTS_OTM_RATIO']
    ] = scaler.fit_transform(
        df[['CALLS_ITM_RATIO', 'CALLS_OTM_RATIO', 'PUTS_ITM_RATIO', 'PUTS_OTM_RATIO']]
        )

# Новые фичи: Отношение OPEN, LOW, HIGH к CLOSE текущего дня
df['OPEN_RATIO'] = df['OPEN'] / df['CLOSE']
df['LOW_RATIO'] = df['LOW'] / df['CLOSE']
df['HIGH_RATIO'] = df['HIGH'] / df['CLOSE']

# Нормализация
df[['OPEN_RATIO', 'LOW_RATIO', 'HIGH_RATIO']] = scaler.fit_transform(
    df[['OPEN_RATIO', 'LOW_RATIO', 'HIGH_RATIO']]
    )

# Новые фичи: Отношение цен за предыдущие 10 дней к CLOSE текущего дня
for col in ['OPEN', 'LOW', 'HIGH', 'CLOSE']:
    for i in range(1, 5):
        df[f'{col}_{i}_RATIO'] = df[col].shift(i) / df['CLOSE']

# Нормализация фичей за 10 дней
cols_to_normalize = (
    [f'{col}_{i}_RATIO' for col in ['OPEN', 'LOW', 'HIGH', 'CLOSE'] for i in range(1, 5)]
    )
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# Создание target -------------------------------------------------------------
# Создание колонки target (бычий или медвежий бар)
df['target'] = (df['OPEN'].shift(-1) < df['CLOSE'].shift(-1)).astype(int)

# Удаление строк с NaN
df = df.dropna()

df = df.drop(
    columns=['TRADEDATE', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLUME', 
             'OPENPOSITION', 'CALLS_ITM', 'CALLS_OTM', 'PUTS_ITM', 'PUTS_OTM']
    )

df.to_csv('features.csv', index=False)

# print(df.columns.tolist())
print(df)
