import os
import json

def most_populous_countries(file_path, top_n=10):

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            sorted_countries = sorted(data, key=lambda x: x.get('population', 0), reverse=True)
            return [(country['name'], country['population']) for country in sorted_countries[:top_n]]
    except FileNotFoundError:
        print(f"Dosya bulunamadı: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"JSON dosyası çözümlenemedi: {file_path}")
        return None

base_path = "File_Handling"

countries_file = os.path.join(base_path, "countries_data.json")
most_populous = most_populous_countries(countries_file)
if most_populous:
    print("En kalabalık ülkeler:")
    for country, population in most_populous:
        print(f"{country}: {population}")