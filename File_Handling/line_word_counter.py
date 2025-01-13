import os

def count_lines_and_words(file_path):

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            line_count = len(lines)
            word_count = sum(len(line.split()) for line in lines)
            return line_count, word_count
    except FileNotFoundError:
        print(f"Dosya bulunamadı: {file_path}")
        return None

base_path = "File_Handling"

dosya_isimleri = [
    "obama_speech.txt",
    "michelle_obama_speech.txt",
    "donald_speech.txt",
    "melina_trump_speech.txt"
]

# Her bir dosyanın satır ve kelime sayısını hesapla
for dosya in dosya_isimleri:
    dosya_yolu = os.path.join(base_path, dosya)
    sonuc = count_lines_and_words(dosya_yolu)
    if sonuc:
        print(f"{dosya}: {sonuc[0]} satır, {sonuc[1]} kelime")