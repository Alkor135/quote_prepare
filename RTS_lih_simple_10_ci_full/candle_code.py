# Исключены дожи. Счинаются как бар на повышение с маленьким телом

from pathlib import Path
import pandas as pd
import sqlite3
import numpy as np
import sys
import os

sys.dont_write_bytecode = True

# Параметры по которым будут производиться расчеты величины теней и тела свечи
PERIOD = 20  # Период для подсчета значений квантилей.
QUAN_MIN = 0.25  # Нижняя граница ниже которой параметр считается маленьким.
QUAN_MAX = 0.75  # Веерхняя граница выше которой параметр считается большим.

def code_binary_to_int():
    """Создание словаря для всех комбинаций строки из 7 символов (0 и 1)"""
    combination_dict = {}  # Создаем пустой словарь

    # Генерируем все комбинации длиной 7 из символов '0' и '1'
    for i in range(2):  # Первый символ
        for j in range(2):  # Второй символ
            for k in range(2):  # Третий символ
                for l in range(2):  # Четвертый символ
                    for m in range(2):  # Пятый символ
                        for n in range(2):  # Шестой символ
                            for o in range(2):  # Седьмой символ
                                # Формируем ключ как строку 'ijklmno'
                                key = f"{i}{j}{k}{l}{m}{n}{o}"
                                # Значение — порядковый номер комбинации
                                value = int(key, 2)  # Перевод из двоичной системы в десятичную
                                # Добавляем комбинацию в словарь
                                combination_dict[key] = value
    return combination_dict

def coding(
        OPEN, CLOSE, size_hi, size_body, size_lo, q_hi_min, q_hi_max, q_body_min, q_body_max, 
        q_lo_min, q_lo_max
        ) -> str:
    """
    Кодирование свечей по Лиховидову
    """
    code_str: str = ''  # Строка в которую будем собирать код свечи
    # Свеча на понижение (медвежья)
    if OPEN > CLOSE:  # Свеча на понижение (медвежья)
        code_str += '0'
        # Для тела медвежьей свечи
        if size_body > q_body_max:  # 00 - медвежья свеча с телом больших размеров
            code_str += '00'
        elif size_body >= q_body_min:  # 01 - медвежья свеча с телом средних размеров
            code_str += '01'
        elif size_body > 0.0:  # 10 - медвежья свеча с телом небольших размеров
            code_str += '10'
        # Для верхней тени медвежьей свечи
        if size_hi > q_hi_max:  # 11 - верхняя тень больших размеров
            code_str += '11'
        elif size_hi >= q_hi_min:  # 10 - верхняя тень средних размеров
            code_str += '10'
        elif size_hi > 0.0:  # 01 - верхняя тень небольших размеров
            code_str += '01'
        else:  # 00 - верхняя тень отсутствует
            code_str += '00'
        # Для нижней тени медвежьей свечи
        if size_lo > q_lo_max:  # 00 - нижняя тень больших размеров
            code_str += '00'
        elif size_lo >= q_lo_min:  # 01 - нижняя тень средних размеров
            code_str += '01'
        elif size_lo > 0.0:  # 10 - нижняя тень небольших размеров
            code_str += '10'
        else:  # 11 - нижняя тень отсутствует
            code_str += '11'
    # Свеча на повышение (бычья)
    elif OPEN < CLOSE:  # Свеча на повышение (бычья)
        code_str += '1'
        # Для тела бычьей свечи
        if size_body > q_body_max:  # 11 - бычья свеча с телом больших размеров.
            code_str += '11'
        elif size_body >= q_body_min:  # 10 - бычья свеча с телом средних размеров
            code_str += '10'
        elif size_body > 0.0:  # 01 - бычья свеча с телом небольших размеров
            code_str += '01'
        # Для верхней тени бычьей свечи
        if size_hi > q_hi_max:  # 11 - верхняя тень больших размеров
            code_str += '11'
        elif size_hi >= q_body_min:  # 10 - верхняя тень средних размеров
            code_str += '10'
        elif size_hi > 0.0:  # 01 - верхняя тень небольших размеров
            code_str += '01'
        else:  # 00 - верхняя тень отсутствует
            code_str += '00'
        # Для нижней тени бычьей свечи
        if size_lo > q_lo_max:  # 00 - нижняя тень больших размеров
            code_str += '00'
        elif size_lo >= q_lo_min:  # 01 - нижняя тень средних размеров
            code_str += '01'
        elif size_lo > 0.0:  # 10 - нижняя тень небольших размеров
            code_str += '10'
        else:  # 11 - нижняя тень отсутствует
            code_str += '11'
    # Дожи
    else:  # Дожи
        if size_hi > size_lo:  # Верхняя тень больше, медвежий дожи
            code_str += '011'
        else:  # Верхняя тень меньше, бычий дожи
            code_str += '100'
        # Для верхней тени дожи
        if size_hi > q_hi_max:  # 11 - верхняя тень больших размеров
            code_str += '11'
        elif size_hi >= q_hi_min:  # 10 - верхняя тень средних размеров
            code_str += '10'
        elif size_hi > 0.0:  # 01 - верхняя тень небольших размеров
            code_str += '01'
        else:  # 00 - верхняя тень отсутствует
            code_str += '00'
        # Для нижней тени дожи
        if size_lo > q_hi_max:  # 00 - нижняя тень больших размеров
            code_str += '00'
        elif size_lo >= q_lo_min:  # 01 - нижняя тень средних размеров
            code_str += '01'
        elif size_lo > 0.0:  # 10 - нижняя тень небольших размеров
            code_str += '10'
        else:  # 11 - нижняя тень отсутствует
            code_str += '11'
    return code_str


def candle_code(db_path, start_date, end_date):
        # Чтение данных по фьючерсам
    query = """
        SELECT TRADEDATE, OPEN, LOW, HIGH, CLOSE 
        FROM Futures 
        WHERE TRADEDATE BETWEEN ? AND ? 
        ORDER BY TRADEDATE
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))

    df[['TRADEDATE']] = df[['TRADEDATE']].apply(pd.to_datetime)
    
    # Подсчет размеров теней и тела свечи
    df['size_hi'] = df.apply(lambda x: abs(x.HIGH - x.OPEN), axis=1)
    df['size_body'] = df.apply(lambda x: abs(x.OPEN - x.CLOSE), axis=1)
    df['size_lo'] = df.apply(lambda x: abs(x.OPEN - x.LOW), axis=1)
    
    # Подсчет и запись в колонки заданных по условию квантилей для теней и тела свечи
    df['q_hi_min'] = df.size_hi.rolling(window=PERIOD).quantile(QUAN_MIN)  # Минимальный заданный квантиль для верхней тени свечи
    df['q_hi_max'] = df.size_hi.rolling(window=PERIOD).quantile(QUAN_MAX)  # Максимальный заданный квантиль для верхней тени свечи
    df['q_body_min'] = df.size_body.rolling(window=PERIOD).quantile(QUAN_MIN)
    df['q_body_max'] = df.size_body.rolling(window=PERIOD).quantile(QUAN_MAX)
    df['q_lo_min'] = df.size_lo.rolling(window=PERIOD).quantile(QUAN_MIN)
    df['q_lo_max'] = df.size_lo.rolling(window=PERIOD).quantile(QUAN_MAX)
    
    df = df.dropna().reset_index(drop=True)
    
    # Добавление колонки кода свечи
    df['CANDLE_CODE'] = df.apply(lambda x: coding(
        x.OPEN,
        x.CLOSE,
        x.size_hi,
        x.size_body,
        x.size_lo,
        x.q_hi_min,
        x.q_hi_max,
        x.q_body_min,
        x.q_body_max,
        x.q_lo_min,
        x.q_lo_max
        ), axis=1)  # Заполняем столбец candle_code
    
    # Преобразуем свечные коды в числовой формат (список уникальных кодов)
    code_to_int_dic = code_binary_to_int()
    df['CANDLE_INT'] = df['CANDLE_CODE'].map(code_to_int_dic)
    # Создание колонок с признаками 'CANDLE_INT' за 20 предыдущих свечей
    for i in range(1, 21):
        df[f'CI_{i}'] = df['CANDLE_INT'].shift(i).astype('Int64')
    # Удаление колонок CANDLE_CODE и CANDLE_INT
    df = df.drop(columns=[
        'size_hi', 'size_body', 'size_lo', 'q_hi_min', 'CANDLE_CODE', 'CANDLE_INT', 
        'q_hi_max', 'q_body_min', 'q_body_max', 'q_lo_min', 'q_lo_max'
        ])

    # 📌 Создание колонки направления.
    df['DIRECTION'] = (df['CLOSE'] > df['OPEN']).astype(int)

    # df['DIRECTION'] = df[['OPEN', 'CLOSE']].apply(lambda x: 1 if (x.CLOSE > x.OPEN) else 0, axis=1)  # Добавление колонки напрвления свечи
    # df = df[['TRADEDATE', 'OPEN', 'LOW', 'HIGH', 'CLOSE', 'CANDLE_CODE', 'CANDLE_INT', 'DIRECTION']]
    df = df.dropna().reset_index(drop=True)
    
    return df

if __name__ == '__main__':
    # Установка рабочей директории в папку, где находится файл скрипта
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    db_path = Path(r'C:\Users\Alkor\gd\data_quote_db\RTS_futures_options_day_2014.db')

    # Переменные с датами
    start_date = '2014-01-01'
    end_date = '2025-03-11'

    df_fut = candle_code(db_path, start_date, end_date)
    print(df_fut)
   