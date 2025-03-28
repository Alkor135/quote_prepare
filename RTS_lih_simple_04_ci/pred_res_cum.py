"""
Для обучения моделей с разными гиперпараметрами. Лиховидов. Бинарка.
С балансировкой классов добавлением рандомных, где нет совпадения по фичам 
с противоположным классом.
Лучшая модель сохраняется по Profit - Loss критерию.
Только тест.
Сохранение дата фрейма в файл для анализа.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import os
from data_read import data_load
import shutil
import sys
sys.dont_write_bytecode = True


# === СОЗДАНИЕ НЕЙРОСЕТИ (LSTM) (ДОЛЖНА СОВПАДАТЬ С ОБУЧЕННОЙ) ===
class CandleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(CandleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
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

db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_day_2014.db')
data_path = Path(fr"pred_res_cum.csv")
df_data = pd.DataFrame()

for counter in range(1, 101):  # ------------------------------------------------------------------
    # Удаляем папку __pycache__ (если она была создана)
    shutil.rmtree('__pycache__', ignore_errors=True)

    # === 1. ЗАГРУЗКА ДАННЫХ ===
    df_fut = data_load(db_path, '2023-01-01', '2025-03-11')

    df_fut = df_fut.dropna().reset_index(drop=True)

    # window_size = 20

    # === 5. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(fr"model\best_model_{counter}.pth")  #
    model = CandleLSTM(vocab_size=27, embedding_dim=8, hidden_dim=32, output_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 6. ПРОГНОЗИРОВАНИЕ ===
    # 1. Определяем фичи
    X_features = df_fut[[f'CI_{i}' for i in range(1, 21)]].values
    X_features = np.array(X_features, dtype=np.int64)  # Привести к числовому типу

    # 2. Преобразуем в тензор (и перемещаем на `device`)
    X_tensor = torch.tensor(X_features, dtype=torch.long).to(device)

    # 3. Получаем предсказания
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()  # Переводим обратно на CPU

    # Заполняем колонку PREDICTION (первые window_size значений - NaN)
    df_fut[f'PRED_{counter}'] = (predictions > 0.5).astype(int)  # Если Sigmoid

    # Выбор строк, где TRADEDATE больше 2024-01-01
    df = df_fut[df_fut['TRADEDATE'] > '2024-01-01'].copy()

    # === 3. РАСЧЁТ РЕЗУЛЬТАТОВ ПРОГНОЗА ===
    def calculate_result(row):
        if pd.isna(row[f"PRED_{counter}"]):  # Если NaN после сдвига
            return 0  # Можно удалить или оставить 0

        true_direction = 1 if row["CLOSE"] > row["OPEN"] else 0
        predicted_direction = row[f"PRED_{counter}"]

        difference = abs(row["CLOSE"] - row["OPEN"])
        return difference if true_direction == predicted_direction else -difference


    df[f"RES_{counter}"] = df.apply(calculate_result, axis=1)
    df[f"CUM_{counter}"] = df[f"RES_{counter}"].cumsum()
    # print('Проверка counter')
    # print(df)

    if counter == 1:
        df_data = df
        # print('Копирование')
    else:
        # Выполняем правое слияние
        # print('Слияние')
        df_data = pd.merge(
            df_data,
            df,
            how='right',  # правое слияние
            on=['TRADEDATE', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'DIRECTION'].extend(
                [f'CI_{i}' for i in range(1, 21)]
                ),
            suffixes=('_x', '_y')  # добавляем суффиксы для различения одинаковых колонок
        )

    # === 4. Сохранение данных в файл ===
    df_data.to_csv(data_path, index=False)
    print(f"✅ Данные сохранены в файл: '{data_path}', {counter=}")
    # print(df_data)
