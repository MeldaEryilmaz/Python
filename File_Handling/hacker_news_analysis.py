import os
import csv

def count_rows_with_keyword(file_path, keyword, case_sensitive=False):

    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                for cell in row:
                    if (case_sensitive and keyword in cell) or (not case_sensitive and keyword.lower() in cell.lower()):
                        count += 1
                        break
    except FileNotFoundError:
        print(f"Dosya bulunamadı: {file_path}")
    return count

def count_rows_exclusive(file_path, include_keyword, exclude_keyword, case_sensitive=False):

    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                contains_include = False
                contains_exclude = False
                for cell in row:
                    if (case_sensitive and include_keyword in cell) or (not case_sensitive and include_keyword.lower() in cell.lower()):
                        contains_include = True
                    if (case_sensitive and exclude_keyword in cell) or (not case_sensitive and exclude_keyword.lower() in cell.lower()):
                        contains_exclude = True
                if contains_include and not contains_exclude:
                    count += 1
    except FileNotFoundError:
        print(f"Dosya bulunamadı: {file_path}")
    return count

file_path = os.path.join("File_Handling", "hacker_news.csv")

python_count = count_rows_with_keyword(file_path, "Python")
print(f"Python içeren satırların sayısı: {python_count}")

javascript_count = count_rows_with_keyword(file_path, "JavaScript") + count_rows_with_keyword(file_path, "javascript")
print(f"JavaScript içeren satırların sayısı: {javascript_count}")

java_exclusive_count = count_rows_exclusive(file_path, "Java", "JavaScript")
print(f"JavaScript değil Java içeren satırların sayısı: {java_exclusive_count}")