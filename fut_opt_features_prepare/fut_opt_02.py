import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

# Преобразуем TRADEDATE в datetime
df_fut['TRADEDATE'] = pd.to_datetime(df_fut['TRADEDATE'])
df_opt['TRADEDATE'] = pd.to_datetime(df_opt['TRADEDATE'])

# Объединяем df_opt с df_fut по TRADEDATE
df_opt = df_opt.merge(df_fut[['TRADEDATE', 'CLOSE']], on='TRADEDATE', how='left')

# Фильтрация и группировка
feature_columns = ['CALLS_ITM', 'CALLS_OTM', 'PUTS_ITM', 'PUTS_OTM']
groups = {
    'CALLS_ITM': ('C', '<'),
    'CALLS_OTM': ('C', '>'),
    'PUTS_ITM': ('P', '<'),
    'PUTS_OTM': ('P', '>')
}

def get_grouped_data(option_type, comparator):
    return df_opt[(df_opt['OPTIONTYPE'] == option_type) & (df_opt['STRIKE'].astype(float) < df_opt['CLOSE'] if comparator == '<' else df_opt['STRIKE'].astype(float) > df_opt['CLOSE'])]

df_grouped = {name: get_grouped_data(opt, comp).groupby('TRADEDATE')['OPENPOSITION'].sum().rename(name) for name, (opt, comp) in groups.items()}

df = df_fut.copy()
for col in feature_columns:
    df = df.merge(df_grouped[col], on='TRADEDATE', how='left')
    df[col].fillna(0, inplace=True)

# Вычисляем разницу в днях до истечения
scaler = MinMaxScaler(feature_range=(0, 1))
df['norm_date_diff'] = scaler.fit_transform(((pd.to_datetime(df['LSTTRADE']) - df['TRADEDATE']).dt.days).values.reshape(-1, 1)).clip(0, 1)
df.drop(columns=['LSTTRADE'], inplace=True)

# Фичи объемов
df['VOLUME_MEAN_10'] = df['VOLUME'].shift(1).rolling(window=10, min_periods=1).mean()
df['VOLUME_RATIO'] = df['VOLUME'] / df['VOLUME_MEAN_10']
df['norm_vol'] = scaler.fit_transform(df[['VOLUME_RATIO']])
df.drop(columns=['VOLUME_MEAN_10', 'VOLUME_RATIO'], inplace=True)

# Отношение CALLS/PUTS к OPENPOSITION
for col in feature_columns:
    df[f'{col}_RATIO'] = df[col] / df['OPENPOSITION']
    df[f'{col}_RATIO'].fillna(0, inplace=True)

df[[f'{col}_RATIO' for col in feature_columns]] = scaler.fit_transform(df[[f'{col}_RATIO' for col in feature_columns]])

# Отношение цен к CLOSE
df['OPEN_RATIO'] = df['OPEN'] / df['CLOSE']
df['LOW_RATIO'] = df['LOW'] / df['CLOSE']
df['HIGH_RATIO'] = df['HIGH'] / df['CLOSE']
df[['OPEN_RATIO', 'LOW_RATIO', 'HIGH_RATIO']] = scaler.fit_transform(df[['OPEN_RATIO', 'LOW_RATIO', 'HIGH_RATIO']])

# Отношение цен за предыдущие 4 дня к CLOSE
for col in ['OPEN', 'LOW', 'HIGH', 'CLOSE']:
    for i in range(1, 5):
        df[f'{col}_{i}_RATIO'] = df[col].shift(i) / df['CLOSE']
df[[f'{col}_{i}_RATIO' for col in ['OPEN', 'LOW', 'HIGH', 'CLOSE'] for i in range(1, 5)]] = scaler.fit_transform(df[[f'{col}_{i}_RATIO' for col in ['OPEN', 'LOW', 'HIGH', 'CLOSE'] for i in range(1, 5)]])

# Добавление фичей отношений EMA между ближайшими периодами
for period in range(20, 40):
    short_ema = df['CLOSE'].ewm(span=period, adjust=False).mean()
    long_ema = df['CLOSE'].ewm(span=period + 1, adjust=False).mean()
    df[f'EMA_RATIO_{period}_{period+1}'] = short_ema / long_ema

df[[f'EMA_RATIO_{period}_{period+1}' for period in range(20, 40)]] = scaler.fit_transform(df[[f'EMA_RATIO_{period}_{period+1}' for period in range(20, 40)]])

# Создание target
df['target'] = (df['OPEN'].shift(-1) < df['CLOSE'].shift(-1)).astype(int)

# Удаление строк с NaN
df.dropna(inplace=True)

df.drop(columns=['TRADEDATE', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'VOLUME', 'OPENPOSITION', 'CALLS_ITM', 'CALLS_OTM', 'PUTS_ITM', 'PUTS_OTM'], inplace=True)

df.to_csv('features.csv', index=False)

print(df)
