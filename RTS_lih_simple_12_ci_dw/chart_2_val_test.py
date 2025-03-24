"""
Сохранение 2 графиков (валидационного и тестового) в файл.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
from data_read import data_load
import shutil
import sys
sys.dont_write_bytecode = True


# === ОПРЕДЕЛЕНИЕ МОДЕЛИ (ДОЛЖНА СОВПАДАТЬ С ОБУЧЕННОЙ) ===
class CandleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, day_vocab_size, day_embedding_dim):
        super(CandleLSTM, self).__init__()
        self.embedding_candle = nn.Embedding(vocab_size, embedding_dim)  # Эмбеддинги для свечных кодов
        self.embedding_day = nn.Embedding(day_vocab_size, day_embedding_dim)  # Эмбеддинги для дней недели
        self.lstm = nn.LSTM(embedding_dim + day_embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_candle, x_day):
        x_candle = self.embedding_candle(x_candle)  # Преобразуем свечные коды в эмбеддинги
        x_day = self.embedding_day(x_day)  # Преобразуем дни недели в эмбеддинги
        x = torch.cat((x_candle, x_day), dim=-1)  # Объединяем эмбеддинги свечей и дней недели
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)

# === РАСЧЁТ РЕЗУЛЬТАТОВ ПОСЛЕ ПРОГНОЗА ===
def calculate_result(row):
    if pd.isna(row["PREDICTION"]):  # Если NaN 
        return 0  # Можно удалить или оставить 0

    true_direction = 1 if row["CLOSE"] > row["OPEN"] else 0
    predicted_direction = row["PREDICTION"]

    difference = abs(row["CLOSE"] - row["OPEN"])
    return difference if true_direction == predicted_direction else -difference

# Установка рабочей директории в папку, где находится файл скрипта
script_dir = Path(__file__).parent
os.chdir(script_dir)

db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day_2014.db')

# Инициализация модели
vocab_size = 27  # Количество уникальных кодов свечей
embedding_dim = 16
day_vocab_size = 7  # Количество уникальных дней недели (0-6)
day_embedding_dim = 4
hidden_dim = 64
output_dim = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for counter in range(1, 101):
    # Удаляем папку __pycache__ (если она была создана)
    shutil.rmtree('__pycache__', ignore_errors=True)
    
    # === ЗАГРУЗКА ДАННЫХ ДЛЯ ВАЛИДАЦИОННОГО ГРАФИКА ===-------------------------------------------
    df_fut = data_load(db_path, '2014-01-01', '2024-01-01')
    df_fut = df_fut.dropna().reset_index(drop=True)

    # === ПОДГОТОВКА ДАННЫХ ДЛЯ ПРЕДСКАЗАНИЯ ===
    feature_candle_columns = [col for col in df_fut.columns if col.startswith('CI_')]
    feature_day_columns = [col for col in df_fut.columns if col.startswith('DAY_W_')]

    X_candle = df_fut[feature_candle_columns].values.astype(np.int64)
    X_day = df_fut[feature_day_columns].values.astype(np.int64)

    X_candle = np.array(X_candle, dtype=np.int64)  # Привести к числовому типу
    X_day = np.array(X_day, dtype=np.int64)  # Привести к числовому типу

    X_candle_tensor = torch.tensor(X_candle, dtype=torch.long).to(device)
    X_day_tensor = torch.tensor(X_day, dtype=torch.long).to(device)

    # === 5. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
    model_path = Path(fr"model\best_model_{counter}.pth")
    model = CandleLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, day_vocab_size, day_embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 6. ПРОГНОЗИРОВАНИЕ ===
    with torch.no_grad():
        predictions = model(X_candle_tensor, X_day_tensor).cpu().numpy()  # Переводим обратно на CPU

    # 4. Если модель бинарная: Преобразуем выходные значения в классы (0 или 1)
    df_fut['PREDICTION'] = (predictions > 0.5).astype(int)  # Если Sigmoid

    split = int(len(df_fut) * 0.85)  # 85% - обучающая выборка, 15% - тестовая
    df_val = df_fut.iloc[split:].copy()  # Берем последние 15%

    # Применяем calculate_result и сохраняем результат в новой колонке
    df_val.loc[:, "RESULT"] = df_val.apply(calculate_result, axis=1)

    # Создаем кумулятивный результат
    df_val.loc[:, "CUMULATIVE_RESULT"] = df_val["RESULT"].cumsum()

    # === ЗАГРУЗКА ДАННЫХ ДЛЯ ТЕСТОВАГО ГРАФИКА ===------------------------------------------------
    df_fut = data_load(db_path, '2023-01-01', '2025-03-11')
    df_fut = df_fut.dropna().reset_index(drop=True)

    # === ПОДГОТОВКА ДАННЫХ ДЛЯ ПРЕДСКАЗАНИЯ ===
    feature_candle_columns = [col for col in df_fut.columns if col.startswith('CI_')]
    feature_day_columns = [col for col in df_fut.columns if col.startswith('DAY_W_')]

    X_candle = df_fut[feature_candle_columns].values.astype(np.int64)
    X_day = df_fut[feature_day_columns].values.astype(np.int64)

    X_candle = np.array(X_candle, dtype=np.int64)  # Привести к числовому типу
    X_day = np.array(X_day, dtype=np.int64)  # Привести к числовому типу

    X_candle_tensor = torch.tensor(X_candle, dtype=torch.long).to(device)
    X_day_tensor = torch.tensor(X_day, dtype=torch.long).to(device)

    # === 5. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
    model_path = Path(fr"model\best_model_{counter}.pth")
    model = CandleLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, day_vocab_size, day_embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 6. ПРОГНОЗИРОВАНИЕ ===
    with torch.no_grad():
        predictions = model(X_candle_tensor, X_day_tensor).cpu().numpy()  # Переводим обратно на CPU

    # 4. Если модель бинарная: Преобразуем выходные значения в классы (0 или 1)
    df_fut['PREDICTION'] = (predictions > 0.5).astype(int)  # Если Sigmoid

    # Выбор строк, где TRADEDATE больше 2024-01-01
    df_test = df_fut.query("'2024-01-01' <= TRADEDATE <= '2025-03-11'").copy()
    # df_test = df_fut[df_fut['TRADEDATE'] > '2024-01-01'].copy()

    # Применяем calculate_result и сохраняем результат в новой колонке
    df_test.loc[:, "RESULT"] = df_test.apply(calculate_result, axis=1)

    # Создаем кумулятивный результат
    df_test.loc[:, "CUMULATIVE_RESULT"] = df_test["RESULT"].cumsum()

    # === СОХРАНЕНИЕ ГРАФИКОВ === -----------------------------------------------------------------
    # === ПОСТРОЕНИЕ КУМУЛЯТИВНОГО ГРАФИКА ===
    # Создание фигуры
    plt.figure(figsize=(14, 12))

    # Первый подграфик
    plt.subplot(2, 1, 1)  # (количество строк, количество столбцов, индекс графика)
    plt.plot(df_val["TRADEDATE"], df_val[f"CUMULATIVE_RESULT"], label="Cumulative Result", 
             color="b")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Result")
    plt.title(f"Валидация Sum RTS. set_seed={counter}")
    plt.legend()
    plt.grid()
    plt.xticks(df_val["TRADEDATE"][::10], rotation=90)

    # Второй подграфик
    plt.subplot(2, 1, 2)  # (количество строк, количество столбцов, индекс графика)
    plt.plot(df_test["TRADEDATE"], df_test[f"CUMULATIVE_RESULT"], label="Cumulative Result", 
             color="b")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Result")
    plt.title(f"Независимый тест Sum RTS. set_seed={counter}")
    plt.legend()
    plt.grid()
    plt.xticks(df_test["TRADEDATE"][::10], rotation=90)

    # Сохранение графика в файл
    plt.tight_layout()
    img_path = Path(fr"chart_2/s_{counter}_RTS.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"✅ График сохранен в файл: '{img_path}'")
    # plt.show()
    plt.close()
    