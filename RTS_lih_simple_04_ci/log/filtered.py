import os
from pathlib import Path

# Установка рабочей директории в папку, где находится файл скрипта
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Путь к файлу
file_path = r"log_model_epoch_seed.txt"

# Список элементов, которые должны быть в строке
required_elements = [
    "Seed=33", 
    "Seed=40", 
    "Seed=3;", 
    "Seed=23", 
    "Seed=87", 
    "Seed=15", 
    "Seed=99", 
    "Seed=83", 
    "Seed=72", 
    "Seed=52"
    ]

# Читаем файл и фильтруем строки
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Оставляем только строки, содержащие хотя бы один элемент из списка
filtered_lines = [line for line in lines if any(element in line for element in required_elements)]

# Перезаписываем файл только с отфильтрованными строками
with open(file_path, "w", encoding="utf-8") as file:
    file.writelines(filtered_lines)
