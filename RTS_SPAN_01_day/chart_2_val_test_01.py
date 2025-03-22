import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import shutil

# Определение модели
class CandleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CandleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    # def forward(self, x):
    #     x, _ = self.lstm(x)
    #     print("Shape before FC:", x.shape)
    #     x = self.fc(x[:, -1, :])
    #     return self.sigmoid(x)

    def forward(self, x_span):
        # Добавляем ось последовательности, если вход 2D (batch_size, feature_dim)
        if len(x_span.shape) == 2:
            x_span = x_span.unsqueeze(1)  # Теперь (batch_size, 1, feature_dim)

        x, _ = self.lstm(x_span)  # Ожидаем (batch_size, seq_len, hidden_dim)

        # Проверяем, что размерность после LSTM корректная
        # print("Shape after LSTM:", x.shape)  # Должно быть (batch_size, seq_len, hidden_dim)

        x = self.fc(x[:, -1, :])  # Берём последний шаг последовательности
        return self.sigmoid(x)

# Функция для расчёта результата
def calculate_result(row):
    if pd.isna(row["PREDICTION"]):
        return 0
    true_direction = 1 if row["CLOSE"] > row["OPEN"] else 0
    return abs(row["CLOSE"] - row["OPEN"]) if true_direction == row["PREDICTION"] else -abs(row["CLOSE"] - row["OPEN"])

# Установка рабочей директории
script_dir = Path(__file__).parent
os.chdir(script_dir)

df = pd.read_csv(r'span_nn_prepare.csv', parse_dates=['TRADEDATE'])

for counter in range(1, 101):
    shutil.rmtree('__pycache__', ignore_errors=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for data_type, start_date, end_date, graph_title in [
        ('val', '2022-09-01', '2024-01-01', "Валидация Sum RTS"),
        ('test', '2024-01-01', '2025-03-10', "Независимый тест Sum RTS")
    ]:
        df_fut = df.query(f"'{start_date}' < TRADEDATE <= '{end_date}'").copy()
        df_fut = df_fut.dropna().reset_index(drop=True)

        # feature_columns = [col for col in df_fut.columns if col.startswith('mp')]
        # X_f = df_fut[feature_columns].values.astype(np.float32)
        X_f = df_fut[[str(i) for i in range(-25000, 25001, 2500)]].values
        X_tensor = torch.tensor(X_f, dtype=torch.float32).unsqueeze(1).to(device)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(fr"model/best_model_{counter}.pth")
        # model = CandleLSTM(input_dim=70, hidden_dim=128, output_dim=1).to(device)
        model = CandleLSTM(input_dim=X_f.shape[1], hidden_dim=32, output_dim=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()
        
        df_fut['PREDICTION'] = (predictions > 0.5).astype(int)
        df_fut["RESULT"] = df_fut.apply(calculate_result, axis=1)
        df_fut["CUMULATIVE_RESULT"] = df_fut["RESULT"].cumsum()

        if data_type == 'val':
            df_val = df_fut
        else:
            df_test = df_fut
    
    # Построение графиков
    plt.figure(figsize=(14, 12))
    
    for idx, (df_plot, title) in enumerate([(df_val, "Валидация Sum RTS"), (df_test, "Независимый тест Sum RTS")], 1):
        plt.subplot(2, 1, idx)
        plt.plot(df_plot["TRADEDATE"], df_plot["CUMULATIVE_RESULT"], label="Cumulative Result", color="b")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Result")
        plt.title(f"{title}. set_seed={counter}")
        plt.legend()
        plt.grid()
        plt.xticks(df_plot["TRADEDATE"][::15 if idx == 1 else 10], rotation=90)
    
    plt.tight_layout()
    img_path = Path(fr"chart_2/s_{counter}_RTS.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"✅ График сохранен в файл: '{img_path}'")
    plt.close()
