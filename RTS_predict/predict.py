"""
Предсказание по моделям.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import os
from data_read import data_load
import shutil
import sys
sys.dont_write_bytecode = True

# === ОПРЕДЕЛЕНИЕ МОДЕЛИ (ДОЛЖНА СОВПАДАТЬ С ОБУЧЕННОЙ) ===
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
    
def predict(model_path, X):
    """ Предсказание по модели """
    try:
        model = CandleLSTM(vocab_size=27, embedding_dim=8, hidden_dim=32, output_dim=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"Ошибка: Файл модели {model_path} не найден.")
        return None

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()  # Переводим обратно на CPU
    
    y = (predictions > 0.5).astype(int)
    return y.item()


if __name__ == '__main__':
    # Установка рабочей директории в папку, где находится файл скрипта
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day_2014.db')
    start_date = '2024-11-01'  # Начальная дата
    model_1_path = Path(fr"model\best_model_3.pth")
    model_2_path = Path(fr"model\best_model_69.pth")

    df = data_load(db_path, start_date)
    df = df.tail(1)  # Взять последнюю запись (для предсказаний)
    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])

    # Определяем фичи
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    X_features = df[[f'CI_{i}' for i in range(1, 21)]].values
    X_features = np.array(X_features, dtype=np.int64)  # Привести к числовому типу
    # Преобразуем в тензор (и перемещаем на `device`)
    X_tensor = torch.tensor(X_features, dtype=torch.long).to(device) 

    # Прогнозирование для каждой модели
    for model_path in [model_1_path, model_2_path]:
        # Удаляем папку __pycache__ (если она была создана)
        shutil.rmtree('__pycache__', ignore_errors=True)

        y_pred = predict(model_path, X_tensor)
        print(f"{df['TRADEDATE'].iloc[-1].date()} {model_path.name}: {y_pred}")
