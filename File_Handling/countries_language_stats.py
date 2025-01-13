import os
import json

def most_spoken_languages(file_path, top_n=10):

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            language_counts = {}
            for country in data:
                for language in country.get('languages', []):
                    language_counts[language] = language_counts.get(language, 0) + 1
            sorted_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
            return sorted_languages[:top_n]
    except FileNotFoundError:
        print(f"Dosya bulunamadı: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"JSON dosyası çözümlenemedi: {file_path}")
        return None

base_path = "File_Handling"

countries_file = os.path.join(base_path, "countries_data.json")
most_spoken = most_spoken_languages(countries_file)
if most_spoken:
    print("En çok konuşulan diller:")
    for language, count in most_spoken:
        print(f"{language}: {count}")