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

# Сохранение словаря в JSON
with open("code_full_int.json", "w") as f:
    json.dump(combination_dict, f)

"""
Использование глобального полного сохраненного словаря.
import json
# Загрузка словаря
with open("code_to_int.json", "r") as f:
    code_to_int = json.load(f)

# Кодируем новые данные, используя старый словарь
df_new['CANDLE_INT'] = df_new['CANDLE_CODE'].map(code_to_int)
"""
