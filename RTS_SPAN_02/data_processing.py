import pandas as pd
from sklearn.utils import resample
import numpy as np
import sys
sys.dont_write_bytecode = True


# === Функция расчета P/L (по предсказанному направлению) ===
def calculate_pnl(y_preds, open_prices, close_prices):
    pnl = 0
    for i in range(len(y_preds)):
        if y_preds[i] > 0.5:  # Покупка (LONG)
            pnl += close_prices[i] - open_prices[i]
        else:  # Продажа (SHORT)
            pnl += open_prices[i] - close_prices[i]
    return pnl  # Итоговая прибыль


def balance_classes(X_0, X_1, X_2, X_3, X_4, X_5, X_6, y):
    """
    Функция для балансировки классов методом Oversampling.
    """
    # Создаем DataFrame
    df_0 = pd.DataFrame(X_0, columns=[f'mp0_{i}' for i in range(1, 11)])
    df_1 = pd.DataFrame(X_1, columns=[f'mp1_{i}' for i in range(1, 11)])
    df_2 = pd.DataFrame(X_2, columns=[f'mp2_{i}' for i in range(1, 11)])
    df_3 = pd.DataFrame(X_3, columns=[f'mp3_{i}' for i in range(1, 11)])
    df_4 = pd.DataFrame(X_4, columns=[f'mp4_{i}' for i in range(1, 11)])
    df_5 = pd.DataFrame(X_5, columns=[f'mp5_{i}' for i in range(1, 11)])
    df_6 = pd.DataFrame(X_6, columns=[f'mp6_{i}' for i in range(1, 11)])
    df_train = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5, df_6], axis=1)

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
    X_0_bal = df_balanced[[f'mp0_{i}' for i in range(1, 11)]].values
    X_1_bal = df_balanced[[f'mp1_{i}' for i in range(1, 11)]].values
    X_2_bal = df_balanced[[f'mp2_{i}' for i in range(1, 11)]].values
    X_3_bal = df_balanced[[f'mp3_{i}' for i in range(1, 11)]].values
    X_4_bal = df_balanced[[f'mp4_{i}' for i in range(1, 11)]].values
    X_5_bal = df_balanced[[f'mp5_{i}' for i in range(1, 11)]].values
    X_6_bal = df_balanced[[f'mp6_{i}' for i in range(1, 11)]].values
    y_bal = df_balanced['TARGET'].values

    # Проверяем новое распределение классов
    print("Распределение после балансировки:\n", pd.Series(y_bal).value_counts())

    return (
        np.array(X_0_bal, dtype=np.float16), 
        np.array(X_1_bal, dtype=np.float16), 
        np.array(X_2_bal, dtype=np.float16), 
        np.array(X_3_bal, dtype=np.float16), 
        np.array(X_4_bal, dtype=np.float16), 
        np.array(X_5_bal, dtype=np.float16), 
        np.array(X_6_bal, dtype=np.float16), 
        np.array(y_bal, dtype=np.int64)
        )
