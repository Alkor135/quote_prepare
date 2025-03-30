import pandas as pd
from sklearn.utils import resample
from pathlib import Path
import numpy as np
import sqlite3
import os
import sys
sys.dont_write_bytecode = True


def balance_classes(X, y, bodies, avg_bodies):
    """
    Функция для балансировки классов методом Oversampling.
    """
    # Создаем DataFrame
    df_train = pd.DataFrame(X, columns=[f'CI_{i}' for i in range(1, 21)])

    # Добавляем остальные колонки
    df_train['TARGET'] = y
    df_train['BODY'] = bodies
    df_train['AVG_BODY'] = avg_bodies

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
    # print(df_balanced)

    # Разделяем обратно
    X_bal = df_balanced[[f'CI_{i}' for i in range(1, 21)]].values
    bodies_bal = df_balanced['BODY'].values
    avg_bodies_bal = df_balanced['AVG_BODY'].values
    y_bal = df_balanced['TARGET'].values

    # Проверяем новое распределение классов
    print("Распределение после балансировки:\n", pd.Series(y_bal).value_counts())

    return (
        np.array(X_bal, dtype=np.int64), 
        np.array(y_bal, dtype=np.int64), 
        np.array(bodies_bal, dtype=np.int64),
        np.array(avg_bodies_bal, dtype=np.int64)
        )


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


# def normalize(series):
#     """ Функция нормализации от 0 до 1 """
#     if len(series) == 0:  # если нет данных, возвращаем NaN
#         return None
#     return (series.iloc[-1] - series.min()) / (series.max() - series.min()) if series.max() != series.min() else 0


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

    # 📌 Создание колонки направления.
    df_fut['DIRECTION'] = (df_fut['CLOSE'] > df_fut['OPEN']).astype(int)

    # 📌 Создание колонки размера тела свечи.
    df_fut['BODY'] = abs(df_fut['CLOSE'] - df_fut['OPEN']).astype(int)

    # 📌 Создание колонки среднего размера тела свечи.
    df_fut['BODY_AVG'] = df_fut['BODY'].rolling(window=20).mean().shift(1)

    df_fut = df_fut.dropna().reset_index(drop=True).copy()

    # Принудительное приведение типов (если есть проблемы)
    for col in [f'CI_{i}' for i in range(1, 21)]:
        df_fut[col] = pd.to_numeric(df_fut[col], errors='coerce').fillna(0).astype(np.int64)

    df_fut['DIRECTION'] = pd.to_numeric(df_fut['DIRECTION'], errors='coerce').fillna(0).astype(np.int64)
    df_fut['BODY'] = pd.to_numeric(df_fut['BODY'], errors='coerce').fillna(0).astype(np.int64)
    df_fut['BODY_AVG'] = pd.to_numeric(df_fut['BODY_AVG'], errors='coerce').fillna(0).astype(np.int64)

    df_fut = df_fut.copy()

    return df_fut

if __name__ == '__main__':
    # Установка рабочей директории в папку, где находится файл скрипта
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_day_2014.db')

    # Переменные с датами
    start_date = '2014-01-01'
    end_date = '2024-01-01'

    df_fut = data_load(db_path, start_date, end_date)
    print(df_fut)
