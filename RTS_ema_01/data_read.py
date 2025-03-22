import pandas as pd
from sklearn.utils import resample
from pathlib import Path
import numpy as np
import sqlite3
import os
import sys
sys.dont_write_bytecode = True


def balance_classes(X, y):
    """
    Функция для балансировки классов методом Oversampling.
    """
    # Создаем DataFrame
    df_train = pd.DataFrame(X, columns=[f'ed_{i}' for i in range(1, 21)])

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
    X_bal = df_balanced[[f'ed_{i}' for i in range(1, 21)]].values
    y_bal = df_balanced['TARGET'].values

    # Проверяем новое распределение классов
    print("Распределение после балансировки:\n", pd.Series(y_bal).value_counts())

    return (
        np.array(X_bal, dtype=np.float32), 
        np.array(y_bal, dtype=np.int64)
        )


def normalize(series):
    """ Мин-макс нормализация с сохранением знака """
    series = np.array(series, dtype=np.float32)  # Приведение к float32
    if series.size == 0 or np.all(series == 0):  # Проверяем пустые или все-нулевые ряды
        return pd.Series(np.zeros_like(series, dtype=np.float32))
    
    min_val = np.min(np.abs(series))
    max_val = np.max(np.abs(series))
    
    if max_val == min_val:
        return pd.Series(np.zeros_like(series, dtype=np.float32))  # Если все значения одинаковые
    
    normalized = (np.abs(series) - min_val) / (max_val - min_val) * np.sign(series)
    
    return pd.Series(normalized.astype(np.float32))  # Возвращаем Series той же длины


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

    # Преобразуем TRADEDATE в datetime
    df_fut['TRADEDATE'] = pd.to_datetime(df_fut['TRADEDATE'])

    # Расчёт EMA_10
    df_fut['EMA_10'] = df_fut['CLOSE'].ewm(span=10, adjust=False).mean()

    # Расчёт EMA_12
    df_fut['EMA_12'] = df_fut['CLOSE'].ewm(span=12, adjust=False).mean()

    # Создание колонки разницы
    df_fut['ema_diff'] = df_fut['EMA_10'] - df_fut['EMA_12']

    # === 📌 1. СОЗДАНИЕ ПРИЗНАКОВ ИЗ ema_diff ===
    # Создание колонок с признаками 'ema_diff' за 20 предыдущих свечей
    for i in range(1, 21):
        df_fut[f'ed_{i}'] = df_fut['ema_diff'].shift(i).astype('Float32')
    # Выбираем только колонки от ed_1 до ed_20
    columns_to_normalize = [f"ed_{i}" for i in range(1, 21)]
    # Применяем нормализацию построчно
    df_fut[columns_to_normalize] = df_fut[columns_to_normalize].apply(normalize, axis=1)

    # Удаление колонок 'EMA_10', 'EMA_12', 'ema_diff'
    df_fut = df_fut.drop(columns=['EMA_10', 'EMA_12', 'ema_diff'])

    # 📌 Создание колонки направления.
    df_fut['DIRECTION'] = (df_fut['CLOSE'] > df_fut['OPEN']).astype(int)

    df_fut = df_fut.dropna().reset_index(drop=True).copy()

    # Принудительное приведение типов (если есть проблемы)
    for col in [f'ed_{i}' for i in range(1, 21)]:
        df_fut[col] = pd.to_numeric(df_fut[col], errors='coerce').fillna(0).astype(np.float32)

    df_fut['DIRECTION'] = pd.to_numeric(df_fut['DIRECTION'], errors='coerce').fillna(0).astype(np.int64)

    df_fut = df_fut.copy()

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
