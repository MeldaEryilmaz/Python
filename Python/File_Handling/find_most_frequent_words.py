import os
from collections import Counter
import string

def find_most_frequent_words(file_path, top_n=10):

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            # Metindeki noktalama işaretlerini kaldır ve kelimeleri küçük harfe çevir
            translator = str.maketrans('', '', string.punctuation)
            text = text.translate(translator).lower()
            words = text.split()
            word_counts = Counter(words)
            return word_counts.most_common(top_n)
    except FileNotFoundError:
        print(f"Dosya bulunamadı: {file_path}")
        return None

base_path = "File_Handling"

konusma_dosyaları = [
    "obama_speech.txt",
    "michelle_obama_speech.txt",
    "donald_speech.txt",
    "melina_trump_speech.txt"
]

# Her bir konuşmada en sık kullanılan kelimeleri bul
for dosya in konusma_dosyaları:
    dosya_yolu = os.path.join(base_path, dosya)
    most_frequent = find_most_frequent_words(dosya_yolu)
    if most_frequent:
        print(f"{dosya} - En sık kullanılan kelimeler:")
        for word, count in most_frequent:
            print(f"{word}: {count}")
        print()