import os
from pathlib import Path

# Установка рабочей директории в папку, где находится файл скрипта
script_dir = Path(__file__).parent
os.chdir(script_dir)

# Путь к файлу
file_path = r"log_model_epoch_seed.txt"

# Список элементов, которые должны быть в строке
required_elements = [
    "Seed=69", 
    "Seed=3;", 
    "Seed=29", 
    "Seed=53", 
    "Seed=4;", 
    "Seed=89", 
    "Seed=62", 
    "Seed=91", 
    "Seed=6;", 
    "Seed=72"
    ]

# Читаем файл и фильтруем строки
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Оставляем только строки, содержащие хотя бы один элемент из списка
filtered_lines = [line for line in lines if any(element in line for element in required_elements)]

# Перезаписываем файл только с отфильтрованными строками
with open(file_path, "w", encoding="utf-8") as file:
    file.writelines(filtered_lines)
