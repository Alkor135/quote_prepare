"""
Создание графика агрегированных данных по сумме P/L от 2 моделей.
"""
import pandas as pd
import matplotlib.pyplot as plt


# === РАСЧЁТ РЕЗУЛЬТАТОВ ПРОГНОЗА ПО 2 МОДЕЛЯМ ===
def calculate_result(row):
    if pd.isna(row[f"PRED_{first_model}"]):  # Если NaN
        return 0  # Можно удалить или оставить 0
    if pd.isna(row[f"PRED_{second_model}"]):  # Если NaN
        return 0  # Можно удалить или оставить 0

    true_direction = 1 if row["CLOSE"] > row["OPEN"] else 0
    predicted_first = row[f"PRED_{first_model}"]
    predicted_second = row[f"PRED_{second_model}"]

    difference = abs(row["CLOSE"] - row["OPEN"])
    if (true_direction == predicted_first) and (predicted_first == predicted_second):
        return difference
    elif(true_direction != predicted_first) and (predicted_first == predicted_second):
        return -difference
    else:
        return 0

# [51, 91] 93, 51, 91, 30, 54
first_model = 51
second_model = 91

# === ЗАГРУЗКА ФАЙЛА ===
df = pd.read_csv(r"pred_res_cum.csv")

df["RESULT"] = df.apply(calculate_result, axis=1)
df["CUMULATIVE_RESULT"] = df["RESULT"].cumsum()

# Подсчет количества значений равных 0.0 в колонке RESULT
count_zeros = (df['RESULT'] == 0.0).sum()

# Подсчет количества положительных и отрицательных значений
num_positive = df[df['RESULT'] > 0].shape[0]
num_negative = df[df['RESULT'] < 0].shape[0]

# Суммы положительных и отрицательных значений
sum_positive = df[df['RESULT'] > 0]['RESULT'].sum()
sum_negative = df[df['RESULT'] < 0]['RESULT'].sum()

# Подсчет количества значений, равных 1.0, в колонках PRED_51 и PRED_91
count_1 = ((df['PRED_51'] == 1.0) & (df['PRED_91'] == 1.0)).sum()
count_0 = ((df['PRED_51'] == 0.0) & (df['PRED_91'] == 0.0)).sum()

print(f"Количество значений равных 0.0: {count_zeros}")
print(f"Количество положительных значений: {num_positive}")
print(f"Количество отрицательных значений: {num_negative}")
print(f"Сумма положительных значений: {sum_positive}")
print(f"Сумма отрицательных значений: {sum_negative}")
print(f"Парных сигналов на покупку: {count_1}")
print(f"Парных сигналов на продажу: {count_0}")

plt.figure(figsize=(12, 6))
plt.plot(df["TRADEDATE"], df["CUMULATIVE_RESULT"], label="Cumulative Result", color="b")
plt.xlabel("Date")
plt.ylabel("Cumulative Result")
plt.title("Cumulative Sum of Prediction combined 2 models")
plt.legend()
plt.grid()

plt.xticks(df["TRADEDATE"][::10], rotation=90)
plt.show()


"""
По данным на 25 февраля 2025 года, стоимость пункта цены фьючерса RTS-3.25 
на индекс РТС (RTSI) — 1,73 рубля.
По данным на 25 февраля 2025 года, для покупки фьючерса RTS-3.25 на индекс 
РТС (RIH5) необходимо иметь на счету гарантийное обеспечение в 38 164,08
"""
df["CUM_RUB"] = df["CUMULATIVE_RESULT"] * 1.73

plt.figure(figsize=(12, 6))
plt.plot(df["TRADEDATE"], df["CUM_RUB"], label="Cumulative Ruble", color="b")
plt.xlabel("Date")
plt.ylabel("Cumulative Ruble")
plt.title("Cumulative Sum of Prediction combined 2 models")
plt.legend()
plt.grid()

plt.xticks(df["TRADEDATE"][::10], rotation=90)
plt.show()
