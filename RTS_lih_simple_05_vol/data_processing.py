import pandas as pd
from sklearn.utils import resample
import numpy as np

def balance_classes(X_candle, X_volume, y):
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
    df_train = pd.concat([df_train, df_volume], axis=1)

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
    y_bal = df_balanced['TARGET'].values

    # Проверяем новое распределение классов
    print("Распределение после балансировки:\n", pd.Series(y_bal).value_counts())

    return np.array(X_candle_bal, dtype=np.int64), np.array(X_volume_bal, dtype=np.float32), np.array(y_bal, dtype=np.int64)

# === ФУНКЦИЯ КОДИРОВАНИЯ СВЕЧЕЙ (ЛИХОВИДОВ) ===
def encode_candle(row):
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
