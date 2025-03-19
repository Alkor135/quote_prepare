import pandas as pd
from sklearn.utils import resample
import numpy as np


# === Функция расчета P/L (по предсказанному направлению) ===
def calculate_pnl(y_preds, open_prices, close_prices):
    pnl = 0
    for i in range(len(y_preds)):
        if y_preds[i] > 0.5:  # Покупка (LONG)
            pnl += close_prices[i] - open_prices[i]
        else:  # Продажа (SHORT)
            pnl += open_prices[i] - close_prices[i]
    return pnl  # Итоговая прибыль


def balance_classes(X_candle, X_volume, X_day, X_dd, X_io, X_c_itm, X_c_otm, X_p_itm, X_p_otm, y):
    """
    Функция для балансировки классов методом Oversampling.
    """
    # Создаем DataFrame
    df_train = pd.DataFrame(X_candle, columns=[f'CI_{i}' for i in range(1, 21)])
    df_volume = pd.DataFrame(X_volume, columns=[f'VOL_{i}' for i in range(1, 21)])
    df_day = pd.DataFrame(X_day, columns=[f'DAY_W_{i}' for i in range(1, 21)])
    df_dd = pd.DataFrame(X_dd, columns=[f'DD_{i}' for i in range(1, 21)])
    df_io = pd.DataFrame(X_io, columns=[f'IO_{i}' for i in range(1, 21)])
    df_c_itm = pd.DataFrame(X_c_itm, columns=[f'C-ITM_{i}' for i in range(1, 21)])
    df_c_otm = pd.DataFrame(X_c_otm, columns=[f'C-OTM_{i}' for i in range(1, 21)])
    df_p_itm = pd.DataFrame(X_p_itm, columns=[f'P-ITM_{i}' for i in range(1, 21)])
    df_p_otm = pd.DataFrame(X_p_otm, columns=[f'P-OTM_{i}' for i in range(1, 21)])
    df_train = pd.concat([df_train, df_volume, df_day, df_dd, df_io, df_c_itm, df_c_otm, df_p_itm, df_p_otm], axis=1)

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
    X_dd_bal = df_balanced[[f'DD_{i}' for i in range(1, 21)]].values
    X_io_bal = df_balanced[[f'IO_{i}' for i in range(1, 21)]].values
    X_c_itm_bal = df_balanced[[f'C-ITM_{i}' for i in range(1, 21)]].values
    X_c_otm_bal = df_balanced[[f'C-OTM_{i}' for i in range(1, 21)]].values
    X_p_itm_bal = df_balanced[[f'P-ITM_{i}' for i in range(1, 21)]].values
    X_p_otm_bal = df_balanced[[f'P-OTM_{i}' for i in range(1, 21)]].values
    y_bal = df_balanced['TARGET'].values

    # Проверяем новое распределение классов
    print("Распределение после балансировки:\n", pd.Series(y_bal).value_counts())

    return (
        np.array(X_candle_bal, dtype=np.int64), 
        np.array(X_volume_bal, dtype=np.float32), 
        np.array(X_day_bal, dtype=np.int64), 
        np.array(X_dd_bal, dtype=np.int64), 
        np.array(X_io_bal, dtype=np.float32), 
        np.array(X_c_itm_bal, dtype=np.float32), 
        np.array(X_c_otm_bal, dtype=np.float32), 
        np.array(X_p_itm_bal, dtype=np.float32), 
        np.array(X_p_otm_bal, dtype=np.float32), 
        np.array(y_bal, dtype=np.int64)
        # np.array(y_bal, dtype=np.float32)
        )
