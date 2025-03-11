import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Функция расчета максимальной просадки
def max_drawdown(cum_returns):
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

# Улучшенная функция расчета CAGR
def cagr(cum_returns):
    if len(cum_returns) < 2 or cum_returns.isna().all():  # Проверка на NaN
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
    # Удаляем NaN значения из доходностей
    daily_returns = daily_returns.dropna()
    if daily_returns.std() > 0:
        excess_returns = daily_returns - risk_free_rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    else:
        return np.nan

# Коэффициент Сортино: Средняя доходность / Downside Std * sqrt(252)
def sortino_ratio(daily_returns, risk_free_rate=0.0, periods_per_year=252):
    # Удаляем NaN значения из доходностей
    daily_returns = daily_returns.dropna()
    if len(daily_returns[daily_returns < 0]) > 0:
        excess_returns = daily_returns - risk_free_rate
        downside_std = excess_returns[excess_returns < 0].std()
        return excess_returns.mean() / downside_std * np.sqrt(periods_per_year) if downside_std > 0 else np.nan
    else:
        return np.nan

# Загрузка данных
df = pd.read_csv("pred_res_cum.csv", index_col=0, parse_dates=True)

# Выбор всех колонок, начинающихся на 'CUM'
df = df.loc[:, df.columns.str.startswith('CUM')]

# Заменяем inf на NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Расчет метрик для каждой модели
results = []

for column in df.columns:
    cum_returns = df[column].dropna()  # Убираем NaN значения

    if cum_returns.empty or cum_returns.nunique() == 1:  # Проверка на недостаточные данные
        print(f"Warning: {column} has insufficient data or constant values.")
        continue

    # Приводим к дневным доходностям
    daily_returns = cum_returns.pct_change().dropna()

    mdd = max_drawdown(cum_returns)
    growth = cagr(cum_returns)
    smoothness = smoothness_r2(cum_returns)
    recovery = recovery_factor(growth, mdd)
    sharpe = sharpe_ratio(daily_returns)
    sortino = sortino_ratio(daily_returns)

    if abs(mdd) > 1e-6:  # Исключаем деление на 0 для MDD
        composite_index = (0.3 * growth / abs(mdd)) + (0.2 * smoothness) + (0.2 * cum_returns.max()) + (0.2 * sharpe) + (0.1 * sortino)
    else:
        composite_index = np.nan  # Если MDD близка к нулю, устанавливаем индекс как NaN

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

df_results = pd.DataFrame(results).sort_values(by="Composite Index", ascending=False)

df_results = df_results.round(4).fillna("-")

print(df_results.to_string(index=False))
