"""
Анализ композитного индекса.
"Max Drawdown": mdd, "CAGR": growth, "Smoothness R²": smoothness, "Recovery Factor": recovery,
"Sharpe Ratio": sharpe, "Sortino Ratio": sortino, "Max Profit": cum_returns.max(),
"Composite Index": composite_index
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
from pathlib import Path


# Установка рабочей директории в папку, где находится файл скрипта
script_dir = Path(__file__).parent
os.chdir(script_dir)

# # Настройки для отображения широкого df pandas
# pd.options.display.width = 1200
# pd.options.display.max_colwidth = 100
# pd.options.display.max_columns = 100
pd.reset_option('^display.', silent=True)


# Функция расчета максимальной просадки (относительной и абсолютной)
def max_drawdown(cum_returns):
    peak = np.maximum.accumulate(cum_returns)
    drawdown = cum_returns - peak  # Абсолютная просадка
    relative_drawdown = drawdown / peak  # Относительная просадка
    return relative_drawdown.min(), drawdown.min()  # Возвращаем обе метрики


# Улучшенная функция CAGR (учитывает даты если есть в индексе)
def cagr(cum_returns):
    start_value = cum_returns.iloc[0]
    end_value = cum_returns.iloc[-1]

    # Проверяем, что начальное и конечное значения больше 0
    if start_value <= 0 or end_value <= 0:
        return -1  # Возвращаем -1, чтобы указать на некорректные данные

    # Рассчитываем количество лет
    if isinstance(cum_returns.index, pd.DatetimeIndex):
        years = (cum_returns.index[-1] - cum_returns.index[0]).days / 365.25
    else:
        years = len(cum_returns) / 252

    # Проверяем, что количество лет больше 0
    if years <= 0:
        return -1

    # Рассчитываем CAGR
    return (end_value / start_value) ** (1 / years) - 1


# Функция расчета R² (плавность кривой)
def smoothness_r2(cum_returns):
    X = np.arange(len(cum_returns)).reshape(-1, 1)  # Временная шкала
    y = cum_returns.values.reshape(-1, 1)  # Кумулятивная доходность
    model = LinearRegression().fit(X, y)  # Линейная регрессия
    r2 = model.score(X, y)  # Коэффициент детерминации R²

    # Если тренд убывающий (конечное значение меньше начального), делаем R² отрицательным
    if cum_returns.iloc[-1] < cum_returns.iloc[0]:
        r2 = -r2

    return r2


# Коэффициент восстановления: CAGR / |Max Drawdown|
def recovery_factor(cagr_value, mdd):
    return cagr_value / abs(mdd) if mdd != 0 else np.nan


# Коэффициент Шарпа: Средняя доходность / Стандартное отклонение * sqrt(252)
def sharpe_ratio(daily_returns, risk_free_rate=0.0, periods_per_year=252):
    # Убираем NaN значения из доходностей
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()

    # Если после удаления NaN остались данные, считаем
    if daily_returns.std() > 0:
        excess_returns = daily_returns - risk_free_rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    else:
        return np.nan


# Коэффициент Сортино: Средняя доходность / Downside Std * sqrt(252)
def sortino_ratio(daily_returns, risk_free_rate=0.0, periods_per_year=252):
    # Убираем NaN значения из доходностей
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()

    # Если после удаления NaN остались данные, считаем
    if len(daily_returns[daily_returns < 0]) > 0:
        excess_returns = daily_returns - risk_free_rate
        downside_std = excess_returns[excess_returns < 0].std()
        return excess_returns.mean() / downside_std * np.sqrt(periods_per_year) if downside_std > 0 else np.nan
    else:
        return np.nan


# Загрузка данных. 'TRADEDATE' в индекс.
df = pd.read_csv("pred_res_cum.csv", index_col=0, parse_dates=True)

df_cum = df.loc[:, df.columns.str.startswith('CUM')]

# Заменяем значения в строке с индексом 0 на 0 для колонок, начинающихся с 'CUM'
# Получаем дату из индекса нулевой строки
first_date = df_cum.index[0]  # Индекс первой строки
# Проверяем, существует ли строка с этой датой
if first_date in df_cum.index:
    # Записываем нули в колонки, начинающиеся с 'CUM'
    df_cum.loc[first_date, df_cum.filter(like='CUM').columns] = 0
else:
    print(f"Дата {first_date} отсутствует в индексе DataFrame.")
# print(df_cum)

# Расчет метрик для каждой модели
results = []

for column in df_cum.columns:
    cum_returns = df_cum[column].dropna()

    if cum_returns.empty or cum_returns.nunique() == 1:
        print(f"Warning: {column} has insufficient data")
        continue

    # Приводим к дневным доходностям
    daily_returns = cum_returns.pct_change().dropna()

    # Расчет метрик
    relative_mdd, absolute_mdd = max_drawdown(cum_returns)  # Получаем обе просадки
    growth = cagr(cum_returns)
    smoothness = smoothness_r2(cum_returns)
    recovery = recovery_factor(growth, relative_mdd)
    sharpe = sharpe_ratio(daily_returns)
    sortino = sortino_ratio(daily_returns)
    profit = cum_returns.iloc[-1] - cum_returns.iloc[0]  # Расчет Profit

    if abs(relative_mdd) > 1e-6:
        composite_index = (0.3 * growth / abs(relative_mdd)) + (0.2 * smoothness) + (
                    0.2 * cum_returns.max()) + (0.2 * sharpe) + (0.1 * sortino)
    else:
        composite_index = None

    results.append({
        "Model": column,
        "Max Drawdown (Relative)": relative_mdd,
        "Max Drawdown (Absolute)": absolute_mdd,
        "CAGR": growth,
        "Smoothness R²": smoothness,
        "Recovery Factor": recovery,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Profit": profit,
        "Composite Index": composite_index
    })

# Вывод отсортированного списка моделей по композитному индексу
# df_results = pd.DataFrame(results).sort_values(by="Composite Index", ascending=False)
# df_results = pd.DataFrame(results).sort_values(by="Profit", ascending=False)
# df_results = pd.DataFrame(results).sort_values(by="Max Drawdown (Absolute)", ascending=False)
# df_results = pd.DataFrame(results).sort_values(by="Sortino Ratio", ascending=False)
df_results = pd.DataFrame(results).sort_values(by="Smoothness R²", ascending=False)

df_results = df_results.round(4).fillna("-")
df_results = df_results.head(10)

# print(df_results.to_string(index=False))
# print(df_results.head(10).to_string(max_rows=10, max_cols=20))
print(df_results)

# Извлекаем числа из колонки 'Model' и преобразуем их в список
numbers = df_results["Model"].str.extract(r'(\d+)')[0].astype(int).tolist()

print(numbers)

# Проверяем наличие колонок перед выборкой
required_columns = [
    'DIRECTION', f'PRED_{numbers[0]}', f'RES_{numbers[0]}', f'PRED_{numbers[1]}', f'RES_{numbers[1]}'
]
missing_columns = [col for col in required_columns if col not in df.columns]

df_tmp = pd.DataFrame()  # Инициализируем пустой DataFrame
if missing_columns:
    print(f"Отсутствующие колонки: {missing_columns}")
else:
    df_tmp = df[required_columns].copy()
    # print(df_tmp)

print(df_tmp)
