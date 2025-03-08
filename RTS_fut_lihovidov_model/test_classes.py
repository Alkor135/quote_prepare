from pathlib import Path
import pandas as pd


# === 1. ЗАГРУЗКА ФАЙЛА ===
predictions_file = Path(r"test_predict.csv")
df = pd.read_csv(predictions_file)

print(df['PREDICTION'].value_counts(normalize=True))  # Относительная частота
