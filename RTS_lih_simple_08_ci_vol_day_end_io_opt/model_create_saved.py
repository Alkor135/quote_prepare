import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import os
from data_read import data_load
from data_processing import balance_classes, calculate_pnl

# === 1. СОЗДАНИЕ НЕЙРОСЕТИ (LSTM) ===
class CandleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, 
                 day_vocab_size, day_embedding_dim, 
                 dd_vocab_size, dd_embedding_dim,  
                 hidden_dim, output_dim):
        super(CandleLSTM, self).__init__()

        # Embedding слои
        self.embedding_candle = nn.Embedding(vocab_size, embedding_dim)  # Свечи
        self.embedding_day = nn.Embedding(day_vocab_size, day_embedding_dim)  # День недели
        self.embedding_dd = nn.Embedding(dd_vocab_size, dd_embedding_dim)  # DD

        # Входной размер = сумма размеров фич
        input_dim = (embedding_dim + 1 + day_embedding_dim + dd_embedding_dim + 1 + 1 + 1 + 1 + 1 + 1)

        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Полносвязный слой
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_candle, x_volume, x_day, x_dd, x_io, x_c_itm, x_c_otm, x_p_itm, x_p_otm):
        x_candle = self.embedding_candle(x_candle)
        x_day = self.embedding_day(x_day)
        x_dd = self.embedding_dd(x_dd)

        print(f"Shapes before concatenation: {x_candle.shape}, {x_volume.shape}, {x_day.shape}, {x_dd.shape}, "
              f"{x_io.shape}, {x_c_itm.shape}, {x_c_otm.shape}, {x_p_itm.shape}, {x_p_otm.shape}")

        # Объединение фичей
        x = torch.cat((x_candle, x_volume.unsqueeze(-1), x_day, x_dd, 
                       x_io.unsqueeze(-1), x_c_itm.unsqueeze(-1), x_c_otm.unsqueeze(-1), 
                       x_p_itm.unsqueeze(-1), x_p_otm.unsqueeze(-1)), dim=-1)

        # LSTM
        x, _ = self.lstm(x)

        # Полносвязный слой
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)

# === 2. СОЗДАНИЕ DATASET ===
class CandlestickDataset(Dataset):
    def __init__(self, X_candle, X_volume, X_day, X_dd, X_io, X_c_itm, X_c_otm, X_p_itm, X_p_otm, y):
        self.X_candle = torch.tensor(X_candle, dtype=torch.long)
        self.X_volume = torch.tensor(X_volume, dtype=torch.float32)
        self.X_day = torch.tensor(X_day, dtype=torch.long)
        self.X_dd = torch.tensor(X_dd, dtype=torch.long)
        self.X_io = torch.tensor(X_io, dtype=torch.float32)
        self.X_c_itm = torch.tensor(X_c_itm, dtype=torch.float32)
        self.X_c_otm = torch.tensor(X_c_otm, dtype=torch.float32)
        self.X_p_itm = torch.tensor(X_p_itm, dtype=torch.float32)
        self.X_p_otm = torch.tensor(X_p_otm, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        # self.y = torch.tensor(y, dtype=torch.int64)
        # self.y = torch.tensor(y.values, dtype=torch.float32)
        # self.y = torch.tensor(y.to_numpy(), dtype=torch.float32)


    def __len__(self):
        return len(self.X_candle)

    def __getitem__(self, idx):
        x_candle = self.X_candle[idx]
        x_volume = self.X_volume[idx]
        x_day = self.X_day[idx]
        x_dd = self.X_dd[idx]
        x_io = self.X_io[idx]
        x_c_itm = self.X_c_itm[idx]
        x_c_otm = self.X_c_otm[idx]
        x_p_itm = self.X_p_itm[idx]
        x_p_otm = self.X_p_otm[idx]
        
        print(f"Dataset sample shapes: {x_candle.shape}, {x_volume.shape}, {x_day.shape}, {x_dd.shape}, {x_io.shape}, {x_c_itm.shape}, {x_c_otm.shape}, {x_p_itm.shape}, {x_p_otm.shape}")
        
        return (x_candle, x_volume, x_day, x_dd, x_io, x_c_itm, x_c_otm, x_p_itm, x_p_otm, self.y[idx])


# === 3. ФИКСАЦИЯ СЛУЧАЙНЫХ ЧИСЕЛ ДЛЯ ДЕТЕРМИНИРОВАННОСТИ ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === 4. ЗАГРУЗКА ДАННЫХ ===
db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day_2014.db')

# Установка рабочей директории в папку, где находится файл скрипта
script_dir = Path(__file__).parent
os.chdir(script_dir)

for counter in range(1, 101):
    set_seed(counter)

    start_date = '2014-01-01'
    end_date = '2024-01-01'
    df_fut = data_load(db_path, start_date, end_date)

    X_candle = df_fut[[f'CI_{i}' for i in range(1, 21)]].values
    X_volume = df_fut[[f'VOL_{i}' for i in range(1, 21)]].values
    X_day = df_fut[[f'DAY_W_{i}' for i in range(1, 21)]].values
    X_dd = df_fut[[f'DD_{i}' for i in range(1, 21)]].values
    X_io = df_fut[[f'IO_{i}' for i in range(1, 21)]].values
    X_c_itm = df_fut[[f'C-ITM_{i}' for i in range(1, 21)]].values
    X_c_otm = df_fut[[f'C-OTM_{i}' for i in range(1, 21)]].values
    X_p_itm = df_fut[[f'P-ITM_{i}' for i in range(1, 21)]].values
    X_p_otm = df_fut[[f'P-OTM_{i}' for i in range(1, 21)]].values
    y = df_fut['DIRECTION'].values  # Преобразуем сразу в numpy

    split = int(0.85 * len(y))
    # X_train_candle, X_train_volume, X_train_day, X_train_dd, X_train_io, X_train_c_itm, X_train_c_otm, X_train_p_itm, X_train_p_otm, y_train = map(
    #     lambda x: x[:split], [X_candle, X_volume, X_day, X_dd, X_io, X_c_itm, X_c_otm, X_p_itm, X_p_otm, y]
    #     )
    X_train_candle, X_train_volume, X_train_day, X_train_dd, X_train_io, X_train_c_itm, X_train_c_otm, X_train_p_itm, X_train_p_otm, y_train = (
        X_candle[:split], X_volume[:split], X_day[:split], X_dd[:split], X_io[:split], X_c_itm[:split], X_c_otm[:split], X_p_itm[:split], X_p_otm[:split], y[:split]
        )
    # X_test_candle, X_test_volume, X_test_day, X_test_dd, X_test_io, X_test_c_itm, X_test_c_otm, X_test_p_itm, X_test_p_otm, y_test = map(
    #     lambda x: x[split:], [X_candle, X_volume, X_day, X_dd, X_io, X_c_itm, X_c_otm, X_p_itm, X_p_otm, y]
    #     )
    X_test_candle, X_test_volume, X_test_day, X_test_dd, X_test_io, X_test_c_itm, X_test_c_otm, X_test_p_itm, X_test_p_otm, y_test = (
        X_candle[split:], X_volume[split:], X_day[split:], X_dd[split:], X_io[split:], X_c_itm[split:], X_c_otm[split:], X_p_itm[split:], X_p_otm[split:], y[split:]
        )

    X_train_candle, X_train_volume, X_train_day, X_train_dd, X_train_io, X_train_c_itm, X_train_c_otm, X_train_p_itm, X_train_p_otm, y_train = balance_classes(
        X_train_candle, X_train_volume, X_train_day, X_train_dd, X_train_io, X_train_c_itm, X_train_c_otm, X_train_p_itm, X_train_p_otm, y_train
    )

    train_dataset = CandlestickDataset(X_train_candle, X_train_volume, X_train_day, X_train_dd, X_train_io, X_train_c_itm, X_train_c_otm, X_train_p_itm, X_train_p_otm, y_train)
    test_dataset = CandlestickDataset(X_test_candle, X_test_volume, X_test_day, X_test_dd, X_test_io, X_test_c_itm, X_test_c_otm, X_test_p_itm, X_test_p_otm, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # === 5. ОБУЧЕНИЕ МОДЕЛИ ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CandleLSTM(
        vocab_size=27, embedding_dim=8, 
        day_vocab_size=7, day_embedding_dim=4, 
        dd_vocab_size=104, dd_embedding_dim=4,   # Увеличено до максимального значения X_dd + 1
        hidden_dim=32, output_dim=1
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_pnl = float('-inf')
    epoch_best_pnl = 0
    model_path = Path(fr"model\best_model_{counter}.pth")
    early_stop_epochs = 200
    epochs_no_improve = 0

    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_candle_batch, X_volume_batch, X_day_batch, X_dd_batch, X_io_batch, X_c_itm_batch, X_c_otm_batch, X_p_itm_batch, X_p_otm_batch, y_batch in train_loader:
            X_candle_batch, X_volume_batch, X_day_batch, X_dd_batch, X_io_batch, X_c_itm_batch, X_c_otm_batch, X_p_itm_batch, X_p_otm_batch, y_batch = (
                X_candle_batch.to(device),
                X_volume_batch.to(device),
                X_day_batch.to(device),
                X_dd_batch.to(device),
                X_io_batch.to(device),
                X_c_itm_batch.to(device),
                X_c_otm_batch.to(device),
                X_p_itm_batch.to(device),
                X_p_otm_batch.to(device),
                y_batch.to(device)
            )

            print(X_candle_batch.shape, X_volume_batch.shape, X_day_batch.shape, X_dd_batch.shape, 
                X_io_batch.shape, X_c_itm_batch.shape, X_c_otm_batch.shape, X_p_itm_batch.shape, X_p_otm_batch.shape)

            optimizer.zero_grad()
            y_pred = model(X_candle_batch, X_volume_batch, X_day_batch, X_dd_batch, X_io_batch, X_c_itm_batch, X_c_otm_batch, X_p_itm_batch, X_p_otm_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # === Проверка на тесте ===
        model.eval()
        y_preds = []
        with torch.no_grad():
            for X_candle_batch, X_volume_batch, X_day_batch, X_dd_batch, X_io_batch, X_c_itm_batch, X_c_otm_batch, X_p_itm_batch, X_p_otm_batch, _ in test_loader:
                y_pred = model(
                    X_candle_batch.to(device), X_volume_batch.to(device), X_day_batch.to(device), X_dd_batch.to(device), X_io_batch.to(device), 
                    X_c_itm_batch.to(device), X_c_otm_batch.to(device), X_p_itm_batch.to(device), X_p_otm_batch.to(device)
                    ).squeeze().cpu().numpy()
                y_preds.extend(y_pred)

        # === Расчет P/L ===
        test_open_prices = df_fut['OPEN'].iloc[split:].values
        test_close_prices = df_fut['CLOSE'].iloc[split:].values
        pnl = calculate_pnl(y_preds, test_open_prices, test_close_prices)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {total_loss / len(train_loader):.5f}, "
            f"P/L: {pnl:.2f}, "
            f"Best P/L: {best_pnl:.2f}, "
            f"Epoch best P/L: {epoch_best_pnl}, "
            f"seed: {counter}"
            )

        if pnl > best_pnl:
            best_pnl = pnl
            epochs_no_improve = 0
            epoch_best_pnl = epoch + 1
            torch.save(model.state_dict(), model_path)
            print(f"✅ Model saved with P/L: {best_pnl:.2f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_epochs:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break

    # === 7. ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ И ФИНАЛЬНЫЙ ТЕСТ ===
    print("\n🔹 Loading best model for final evaluation...")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_preds_final = []
    with torch.no_grad():
        for X_candle_batch, X_volume_batch, X_day_batch, X_dd_batch, X_io_batch, X_c_itm_batch, X_c_otm_batch, X_p_itm_batch, X_p_otm_batch, _ in test_loader:
            X_candle_batch, X_volume_batch, X_day_batch, X_dd_batch, X_io_batch, X_c_itm_batch, X_c_otm_batch, X_p_itm_batch, X_p_otm_batch = (
                X_candle_batch.to(device), X_volume_batch.to(device), X_day_batch.to(device), X_dd_batch.to(device), X_io_batch.to(device),
                X_c_itm_batch.to(device), X_c_otm_batch.to(device), X_p_itm_batch.to(device), X_p_otm_batch.to(device)
                )
            y_pred = model(
                X_candle_batch, X_volume_batch, X_day_batch, X_dd_batch, X_io_batch, X_c_itm_batch, X_c_otm_batch, X_p_itm_batch, X_p_otm_batch
                ).squeeze().cpu().numpy()
            y_preds_final.extend(y_pred)

    final_pnl = calculate_pnl(y_preds_final, test_open_prices, test_close_prices)
    print(f"🏆 Финальный тест P/L: {final_pnl:.2f}\n")
