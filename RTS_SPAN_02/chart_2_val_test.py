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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CandleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Используем выход последнего таймстепа
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
df = data_load(db_path, '2014-01-01', '2025-03-10')
df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])

for counter in range(1, 101):
    # Удаляем папку __pycache__ (если она была создана)
    shutil.rmtree('__pycache__', ignore_errors=True)
    
    # === ЗАГРУЗКА ДАННЫХ ДЛЯ ВАЛИДАЦИОННОГО ГРАФИКА ===-------------------------------------------
    df_fut = df.query("'2014-01-01' < TRADEDATE < '2024-01-01'").copy()

    df_fut = df_fut.dropna().reset_index(drop=True)

    # === 5. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(fr"model\best_model_{counter}.pth")
    model = CandleLSTM(input_dim=10, hidden_dim=64, output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 6. ПРОГНОЗИРОВАНИЕ ===
    # 1. Определяем фичи
    feature_columns_0 = [col for col in df_fut.columns if col.startswith('mp0')]
    feature_columns_1 = [col for col in df_fut.columns if col.startswith('mp1')]
    feature_columns_2 = [col for col in df_fut.columns if col.startswith('mp2')]
    feature_columns_3 = [col for col in df_fut.columns if col.startswith('mp3')]
    feature_columns_4 = [col for col in df_fut.columns if col.startswith('mp4')]
    feature_columns_5 = [col for col in df_fut.columns if col.startswith('mp5')]
    feature_columns_6 = [col for col in df_fut.columns if col.startswith('mp6')]

    X_f_0 = df_fut[feature_columns_0].values.astype(np.float32)
    X_f_1 = df_fut[feature_columns_1].values.astype(np.float32)
    X_f_2 = df_fut[feature_columns_2].values.astype(np.float32)
    X_f_3 = df_fut[feature_columns_3].values.astype(np.float32)
    X_f_4 = df_fut[feature_columns_4].values.astype(np.float32)
    X_f_5 = df_fut[feature_columns_5].values.astype(np.float32)
    X_f_6 = df_fut[feature_columns_6].values.astype(np.float32)

    # 2. Преобразуем в тензор (и перемещаем на `device`)
    X_tensor_0 = torch.tensor(X_f_0, dtype=torch.float32).to(device)
    X_tensor_1 = torch.tensor(X_f_1, dtype=torch.float32).to(device)
    X_tensor_2 = torch.tensor(X_f_2, dtype=torch.float32).to(device)
    X_tensor_3 = torch.tensor(X_f_3, dtype=torch.float32).to(device)
    X_tensor_4 = torch.tensor(X_f_4, dtype=torch.float32).to(device)
    X_tensor_5 = torch.tensor(X_f_5, dtype=torch.float32).to(device)
    X_tensor_6 = torch.tensor(X_f_6, dtype=torch.float32).to(device)

    # 3. Получаем предсказания
    with torch.no_grad():
        predictions = model(
            X_tensor_0, X_tensor_1, X_tensor_2, X_tensor_3, X_tensor_4, X_tensor_5, X_tensor_6 
            ).cpu().numpy()  # Переводим обратно на CPU

    # 4. Если модель бинарная: Преобразуем выходные значения в классы (0 или 1)
    df_fut['PREDICTION'] = (predictions > 0.5).astype(int)  # Если Sigmoid

    split = int(len(df_fut) * 0.85)  # 85% - обучающая выборка, 15% - тестовая
    df_val = df_fut.iloc[split:].copy()  # Берем последние 15%

    df_val["RESULT"] = df_val.apply(calculate_result, axis=1)

    # === СОЗДАНИЕ КОЛОНКИ КОМУЛЯТИВНОГО РЕЗУЛЬТАТА ===
    df_val["CUMULATIVE_RESULT"] = df_val["RESULT"].cumsum()

    # === ЗАГРУЗКА ДАННЫХ ДЛЯ ТЕСТОВАГО ГРАФИКА ===------------------------------------------------
    df_fut = df.query("'2023-01-01' < TRADEDATE <= '2025-03-10'").copy()

    df_fut = df_fut.dropna().reset_index(drop=True)

    # === 5. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(fr"model\best_model_{counter}.pth")
    model = CandleLSTM(input_dim=10, hidden_dim=64, output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 6. ПРОГНОЗИРОВАНИЕ ===
    # 1. Определяем фичи
    feature_columns_0 = [col for col in df_fut.columns if col.startswith('mp0')]
    feature_columns_1 = [col for col in df_fut.columns if col.startswith('mp1')]
    feature_columns_2 = [col for col in df_fut.columns if col.startswith('mp2')]
    feature_columns_3 = [col for col in df_fut.columns if col.startswith('mp3')]
    feature_columns_4 = [col for col in df_fut.columns if col.startswith('mp4')]
    feature_columns_5 = [col for col in df_fut.columns if col.startswith('mp5')]
    feature_columns_6 = [col for col in df_fut.columns if col.startswith('mp6')]

    X_f_0 = df_fut[feature_columns_0].values.astype(np.float32)
    X_f_1 = df_fut[feature_columns_1].values.astype(np.float32)
    X_f_2 = df_fut[feature_columns_2].values.astype(np.float32)
    X_f_3 = df_fut[feature_columns_3].values.astype(np.float32)
    X_f_4 = df_fut[feature_columns_4].values.astype(np.float32)
    X_f_5 = df_fut[feature_columns_5].values.astype(np.float32)
    X_f_6 = df_fut[feature_columns_6].values.astype(np.float32)

    # 2. Преобразуем в тензор (и перемещаем на `device`)
    X_tensor_0 = torch.tensor(X_f_0, dtype=torch.float32).to(device)
    X_tensor_1 = torch.tensor(X_f_1, dtype=torch.float32).to(device)
    X_tensor_2 = torch.tensor(X_f_2, dtype=torch.float32).to(device)
    X_tensor_3 = torch.tensor(X_f_3, dtype=torch.float32).to(device)
    X_tensor_4 = torch.tensor(X_f_4, dtype=torch.float32).to(device)
    X_tensor_5 = torch.tensor(X_f_5, dtype=torch.float32).to(device)
    X_tensor_6 = torch.tensor(X_f_6, dtype=torch.float32).to(device)

    # 3. Получаем предсказания
    with torch.no_grad():
        predictions = model(
            X_tensor_0, X_tensor_1, X_tensor_2, X_tensor_3, X_tensor_4, X_tensor_5, X_tensor_6 
            ).cpu().numpy()  # Переводим обратно на CPU

    # 4. Если модель бинарная: Преобразуем выходные значения в классы (0 или 1)
    df_fut['PREDICTION'] = (predictions > 0.5).astype(int)  # Если Sigmoid

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
    