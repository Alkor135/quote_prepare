import pandas as pd

# Пример данных доходности для трех графиков
data = {
    'График 1': [100, 120, 150, 130, 180, 170],
    'График 2': [100, 90, 110, 105, 130, 125],
    'График 3': [100, 105, 110, 100, 120, 150]
}
df = pd.DataFrame(data)

# Функция для расчета максимальной прибыли и относительной просадки
def calc_profit_and_drawdown(series):
    max_profit = series.iloc[-1] - series.iloc[0]
    max_drawdown = ((series.cummax() - series) / series.cummax()).max() * 100
    return max_profit, max_drawdown

# Расчет показателей для каждого графика
results = {}
for col in df.columns:
    profit, drawdown = calc_profit_and_drawdown(df[col])
    results[col] = {'Максимальная прибыль': profit, 'Относительная просадка (%)': drawdown}

# Вывод результатов
for key, value in results.items():
    print(f"{key}: Максимальная прибыль = {value['Максимальная прибыль']}, Относительная просадка = {value['Относительная просадка (%)']:.2f}%")

# Примечание: Весовые коэффициенты можно настроить для создания композитного индекса.
