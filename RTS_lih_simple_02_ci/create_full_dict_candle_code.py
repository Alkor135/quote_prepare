"""
Создание полного словаря комбинаций кодов свечей.
"""
import os
import json
from pathlib import Path


# Установка рабочей директории в папку, где находится файл скрипта
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Создаем пустой словарь
combination_dict = {}

# Генерируем все комбинации от 000 до 222
for i in range(3):  # Первое число
    for j in range(3):  # Второе число
        for k in range(3):  # Третье число
            # Формируем ключ в виде строки 'ijk'
            key = f"{i}{j}{k}"
            # Значение — порядковый номер комбинации
            value = i * 9 + j * 3 + k  # Вычисляем индекс комбинации
            # Добавляем в словарь
            combination_dict[key] = value

# Печатаем словарь
for key, value in combination_dict.items():
    print(f"'{key}': {value}")

# 3. Сохранение словаря в JSON
with open("code_full_int.json", "w") as f:
    json.dump(combination_dict, f)

"""
Использование глобального полного сохраненного словаря.
import json
# Загрузка полного словаря
with open("code_full_int.json", "r") as f:
    code_to_int = json.load(f)

# Кодируем данные, используя полный словарь
df_fut['CANDLE_INT'] = df_fut['CANDLE_CODE'].map(code_to_int)
"""
