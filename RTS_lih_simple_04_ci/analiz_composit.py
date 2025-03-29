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


# Функция расчета максимальной просадки
def max_drawdown(cum_returns):
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()


# Улучшенная функция CAGR (учитывает даты если есть в индексе)
def cagr(cum_returns):
    if len(cum_returns) < 2:
        return np.nan

    start_value = cum_returns.iloc[0]
    end_value = cum_returns.iloc[-1]

    if start_value <= 0 or end_value <= 0:
        return np.nan

    if isinstance(cum_returns.index, pd.DatetimeIndex):
        years = (cum_returns.index[-1] - cum_returns.index[0]).days / 365.25
    else:
        years = len(cum_returns) / 252

    return (end_value / start_value) ** (1 / years) - 1


# Функция расчета R² (плавность кривой)
def smoothness_r2(cum_returns):
    X = np.arange(len(cum_returns)).reshape(-1, 1)
    y = cum_returns.values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    return model.score(X, y)


# Коэффициент восстановления: CAGR / |Max Drawdown|
def recovery_factor(cagr_value, mdd):
    return cagr_value / abs(mdd) if mdd != 0 else np.nan


# Коэффициент Шарпа: Средняя доходность / Стандартное отклонение * sqrt(252)
def sharpe_ratio(daily_returns, risk_free_rate=0.0, periods_per_year=252):
    # Убираем NaN значения из доходностей
    daily_returns = daily_returns.dropna()

    # Если после удаления NaN остались данные, считаем
    if daily_returns.std() > 0:
        excess_returns = daily_returns - risk_free_rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    else:
        return np.nan


# Коэффициент Сортино: Средняя доходность / Downside Std * sqrt(252)
def sortino_ratio(daily_returns, risk_free_rate=0.0, periods_per_year=252):
    # Убираем NaN значения из доходностей
    daily_returns = daily_returns.dropna()

    # Если после удаления NaN остались данные, считаем
    if len(daily_returns[daily_returns < 0]) > 0:
        excess_returns = daily_returns - risk_free_rate
        downside_std = excess_returns[excess_returns < 0].std()
        return excess_returns.mean() / downside_std * np.sqrt(
            periods_per_year) if downside_std > 0 else np.nan
    else:
        return np.nan


# Загрузка данных. 'TRADEDATE' в индекс.
df = pd.read_csv("pred_res_cum.csv", index_col=0, parse_dates=True)

df = df.loc[:, df.columns.str.startswith('CUM')]

# Расчет метрик для каждой модели
results = []

for column in df.columns:
    cum_returns = df[column].dropna()

    if cum_returns.empty or cum_returns.nunique() == 1:
        print(f"Warning: {column} has insufficient data")
        continue

    # Приводим к дневным доходностям
    daily_returns = cum_returns.pct_change().dropna()

    mdd = max_drawdown(cum_returns)
    growth = cagr(cum_returns)
    smoothness = smoothness_r2(cum_returns)
    recovery = recovery_factor(growth, mdd)
    sharpe = sharpe_ratio(daily_returns)
    sortino = sortino_ratio(daily_returns)

    if abs(mdd) > 1e-6:
        composite_index = (0.3 * growth / abs(mdd)) + (0.2 * smoothness) + (
                    0.2 * cum_returns.max()) + (0.2 * sharpe) + (0.1 * sortino)
    else:
        composite_index = None

    results.append({
        "Model": column,
        "Max Drawdown": mdd,
        "CAGR": growth,
        "Smoothness R²": smoothness,
        "Recovery Factor": recovery,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Profit": cum_returns.max(),
        "Composite Index": composite_index
    })

# Вывод отсортированного списка моделей по композитному индексу
# df_results = pd.DataFrame(results).sort_values(by="Composite Index", ascending=False)
df_results = pd.DataFrame(results).sort_values(by="Max Profit", ascending=False)

df_results = df_results.round(4).fillna("-")

# print(df_results.to_string(index=False))
print(df_results.head(20).to_string(max_rows=20, max_cols=20))
