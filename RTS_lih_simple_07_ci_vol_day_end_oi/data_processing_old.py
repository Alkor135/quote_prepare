import pandas as pd
from sklearn.utils import resample
from pathlib import Path
import numpy as np
import sqlite3
import os


def code_int():
    # Создаем пустой словарь
    combination_dict = {}

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


def balance_classes(X_candle, X_volume, X_day, y):
    """
    Функция для балансировки классов методом Oversampling.
    
    Args:
        X_candle (np.array): Фичи свечей (категориальные).
        X_volume (np.array): Фичи объемов (числовые).
        y (np.array): Целевая переменная (классы).

    Returns:
        X_candle_bal (np.array): Сбалансированные свечные фичи.
        X_volume_bal (np.array): Сбалансированные объемные фичи.
        y_bal (np.array): Сбалансированные целевые метки.
    """
    # Создаем DataFrame
    df_train = pd.DataFrame(X_candle, columns=[f'CI_{i}' for i in range(1, 21)])
    df_volume = pd.DataFrame(X_volume, columns=[f'VOL_{i}' for i in range(1, 21)])
    df_day = pd.DataFrame(X_day, columns=[f'DAY_W_{i}' for i in range(1, 21)])
    df_train = pd.concat([df_train, df_volume, df_day], axis=1)

    # Добавляем целевую переменную
    df_train['TARGET'] = y

    # Подсчитываем количество примеров в каждом классе
    class_counts = df_train['TARGET'].value_counts()
    print("Распределение перед балансировкой:\n", class_counts)

    # Определяем количество классов
    class_counts = df_train['TARGET'].value_counts()
    min_class = class_counts.idxmin()
    max_class = class_counts.idxmax()

    # Разделяем по классам
    df_minority = df_train[df_train['TARGET'] == min_class]
    df_majority = df_train[df_train['TARGET'] == max_class]

    # Убираем дубликаты редкого класса, которые есть в мажоритарном
    df_minority_unique = df_minority.loc[
        ~df_minority.drop(columns=['TARGET']).apply(tuple, axis=1).isin(
            df_majority.drop(columns=['TARGET']).apply(tuple, axis=1)
        )
    ]

    # Дублируем редкие примеры
    df_minority_upsampled = resample(
        df_minority_unique,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    # Объединяем и перемешиваем
    df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42)

    # Разделяем обратно
    X_candle_bal = df_balanced[[f'CI_{i}' for i in range(1, 21)]].values
    X_volume_bal = df_balanced[[f'VOL_{i}' for i in range(1, 21)]].values
    X_day_bal = df_balanced[[f'DAY_W_{i}' for i in range(1, 21)]].values
    y_bal = df_balanced['TARGET'].values

    # Проверяем новое распределение классов
    print("Распределение после балансировки:\n", pd.Series(y_bal).value_counts())

    return (
        np.array(X_candle_bal, dtype=np.int64), 
        np.array(X_volume_bal, dtype=np.float32), 
        np.array(X_day_bal, dtype=np.float32), 
        np.array(y_bal, dtype=np.int64)
        )

def encode_candle(row):
    # === ФУНКЦИЯ КОДИРОВАНИЯ СВЕЧЕЙ (ЛИХОВИДОВ) ===
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

# === Функция расчета P/L (по предсказанному направлению) ===
def calculate_pnl(y_preds, open_prices, close_prices):
    pnl = 0
    for i in range(len(y_preds)):
        if y_preds[i] > 0.5:  # Покупка (LONG)
            pnl += close_prices[i] - open_prices[i]
        else:  # Продажа (SHORT)
            pnl += open_prices[i] - close_prices[i]
    return pnl  # Итоговая прибыль


# # Функция нормализации от 0 до 1 (для )
# def normalize(values):
#     min_val = values.min()
#     max_val = values.max()
#     return (values - min_val) / (max_val - min_val) if max_val != min_val else values * 0

# Функция нормализации от 0 до 1 (для )
def normalize(series):
    if len(series) == 0:  # если нет данных, возвращаем NaN
        return None
    return (series.iloc[-1] - series.min()) / (series.max() - series.min()) if series.max() != series.min() else 0

def data_prepare(df_fut):
    # Создание кодов свечей по Лиховидову
    df_fut['CANDLE_CODE'] = df_fut.apply(encode_candle, axis=1)

    # Преобразуем свечные коды в числовой формат (список уникальных кодов)
    code_to_int_dic = code_int()
    df_fut['CANDLE_INT'] = df_fut['CANDLE_CODE'].map(code_to_int_dic)

    # Создание колонки направления.
    df_fut['DIRECTION'] = (df_fut['CLOSE'] > df_fut['OPEN']).astype(int)

    # Создание колонок с признаками 'CANDLE_INT' за 20 предыдущих свечей
    for i in range(1, 21):
        df_fut[f'CI_{i}'] = df_fut['CANDLE_INT'].shift(i).astype('Int64')

    # # Создание новой колонки с нормированными значениями
    # df_fut['VOL_NORM'] = (
    #     df_fut['VOLUME']
    #     .shift(1)  # Исключаем текущее значение
    #     .rolling(window=20)  # Смотрим на 20 предыдущих значений
    #     .apply(lambda x: normalize(x), raw=False)  # Применяем нормализацию
    # )

    # Создаем колонку VOL-NORM
    df_fut['VOL-NORM'] = df_fut['VOLUME'].shift(1).rolling(window=20, min_periods=1).apply(normalize, raw=False)
    # Создание колонок с объемом за 20 предыдущих свечей
    for i in range(1, 21):
        df_fut[f'VOL_{i}'] = df_fut['VOL-NORM'].shift(i)  # .astype('Int64')

    # Преобразование колонки TRADEDATE в формат datetime
    df_fut['TRADEDATE'] = pd.to_datetime(df_fut['TRADEDATE'])
    # Создание новой колонки DAY_W с номером дня недели
    df_fut['DAY_W'] = df_fut['TRADEDATE'].dt.weekday
    # Создание колонок с днями за 20 предыдущих свечей
    for i in range(1, 21):
        df_fut[f'DAY_W_{i}'] = df_fut['DAY_W'].shift(i).astype('Int64')

    # Проверка на пропущенные значения
    # print("Количество NaN в df_fut:")
    # print(df_fut.isna().sum())
    df_fut.fillna(0, inplace=True)

    # Проверка типов данных
    # print("\nТипы данных в df_fut:")
    # print(df_fut.dtypes)
    # Принудительное приведение типов (если есть проблемы)
    for col in [f'CI_{i}' for i in range(1, 21)]:
        df_fut[col] = pd.to_numeric(df_fut[col], errors='coerce').fillna(0).astype(np.int64)

    for col in [f'VOL_{i}' for i in range(1, 21)]:
        # df_fut[col] = pd.to_numeric(df_fut[col], errors='coerce').fillna(0).astype(np.float32)
        df_fut[col] = pd.to_numeric(df_fut[col], errors='coerce').fillna(0).astype(np.int64)

    for col in [f'DAY_W_{i}' for i in range(1, 21)]:
        # df_fut[col] = pd.to_numeric(df_fut[col], errors='coerce').fillna(0).astype(np.float32)
        df_fut[col] = pd.to_numeric(df_fut[col], errors='coerce').fillna(0).astype(np.int64)

    df_fut['DIRECTION'] = pd.to_numeric(df_fut['DIRECTION'], errors='coerce').fillna(0).astype(np.int64)

    df_fut = df_fut.dropna().reset_index(drop=True)

    return df_fut

if __name__ == '__main__':
    # === 1. ОПРЕДЕЛЕНИЯ ===
    # Установка рабочей директории в папку, где находится файл скрипта
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_day_full.db')

    # === 2. ЗАГРУЗКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ И ВАЛИДАЦИИ ===
    with sqlite3.connect(db_path) as conn:
        df_fut = pd.read_sql_query(
            """
            SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME 
            FROM Day 
            WHERE TRADEDATE BETWEEN '2014-01-01' AND '2024-01-01' 
            ORDER BY TRADEDATE
            """,
            conn
        )

    df_fut = data_prepare(df_fut)
    print(df_fut)