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
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output

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

for counter in range(1, 101):
    # Удаляем папку __pycache__ (если она была создана)
    shutil.rmtree('__pycache__', ignore_errors=True)
    
    # === ЗАГРУЗКА ДАННЫХ ДЛЯ ВАЛИДАЦИОННОГО ГРАФИКА ===-------------------------------------------
    df_fut = data_load(db_path, '2014-01-01', '2024-01-01')

    df_fut = df_fut.dropna().reset_index(drop=True)

    # === 5. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(fr"model\best_lstm_model_{counter}.pth")
    model = LSTMModel(1, 64, 2, 2).to(device)
    # model = CandleLSTM(vocab_size=27, embedding_dim=8, hidden_dim=32, output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 6. ПРОГНОЗИРОВАНИЕ ===
    # 1. Определяем фичи
    X_features = df_fut[[f'ed_{i}' for i in range(1, 21)]].values
    X_features = np.array(X_features, dtype=np.float32)  # Привести к числовому типу

    # # 2. Преобразуем в тензор (и перемещаем на `device`)
    # X_tensor = torch.tensor(X_features, dtype=torch.float32).to(device)
    # 🔥 Изменяем форму данных: (N, 20) → (N, 20, 1), чтобы соответствовать обученной модели
    X_tensor = torch.tensor(X_features, dtype=torch.float32).unsqueeze(-1).to(device)

    # 3. Получаем предсказания
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()  # Переводим обратно на CPU

    # 4. Если модель бинарная: Преобразуем выходные значения в классы (0 или 1)
    # df_fut['PREDICTION'] = (predictions > 0.5).astype(int)  # Если Sigmoid
    df_fut['PREDICTION'] = predictions.argmax(axis=1)  # Выбираем индекс с наибольшей вероятностью (0 или 1)


    split = int(len(df_fut) * 0.85)  # 85% - обучающая выборка, 15% - тестовая
    df_val = df_fut.iloc[split:].copy()  # Берем последние 15%

    df_val["RESULT"] = df_val.apply(calculate_result, axis=1)

    # === СОЗДАНИЕ КОЛОНКИ КОМУЛЯТИВНОГО РЕЗУЛЬТАТА ===
    df_val["CUMULATIVE_RESULT"] = df_val["RESULT"].cumsum()

    # === ЗАГРУЗКА ДАННЫХ ДЛЯ ТЕСТОВАГО ГРАФИКА ===------------------------------------------------
    df_fut = data_load(db_path, '2023-01-01', '2025-03-11')

    df_fut = df_fut.dropna().reset_index(drop=True)

    # === 5. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(fr"model\best_lstm_model_{counter}.pth")
    # model = CandleLSTM(vocab_size=27, embedding_dim=8, hidden_dim=32, output_dim=1).to(device)
    model = LSTMModel(1, 64, 2, 2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 6. ПРОГНОЗИРОВАНИЕ ===
    # 1. Определяем фичи
    X_features = df_fut[[f'ed_{i}' for i in range(1, 21)]].values
    X_features = np.array(X_features, dtype=np.float32)  # Привести к числовому типу

    # # 2. Преобразуем в тензор (и перемещаем на `device`)
    # X_tensor = torch.tensor(X_features, dtype=torch.float32).to(device)
    # 🔥 Изменяем форму данных: (N, 20) → (N, 20, 1), чтобы соответствовать обученной модели
    X_tensor = torch.tensor(X_features, dtype=torch.float32).unsqueeze(-1).to(device)

    # 3. Получаем предсказания
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()  # Переводим обратно на CPU

    # 4. Если модель бинарная: Преобразуем выходные значения в классы (0 или 1)
    # df_fut['PREDICTION'] = (predictions > 0.5).astype(int)  # Если Sigmoid
    df_fut['PREDICTION'] = predictions.argmax(axis=1)  # Выбираем индекс с наибольшей вероятностью (0 или 1)


    # Выбор строк, где TRADEDATE больше 2024-01-01
    df_test = df_fut[df_fut['TRADEDATE'] > '2024-01-01'].copy()

    df_test["RESULT"] = df_test.apply(calculate_result, axis=1)

    # === 4. ПОСТРОЕНИЕ КУМУЛЯТИВНОГО ГРАФИКА ===
    df_test["CUMULATIVE_RESULT"] = df_test["RESULT"].cumsum()

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
    plt.xticks(df_val["TRADEDATE"][::15], rotation=90)

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
    