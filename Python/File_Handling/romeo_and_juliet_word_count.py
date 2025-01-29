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

file_path = os.path.join("File_Handling", "romeo_and_juliet.txt")

most_frequent_words = find_most_frequent_words(file_path)
if most_frequent_words:
    print("Romeo and Juliet metnindeki en sık kullanılan kelimeler:")
    for word, count in most_frequent_words:
        print(f"{word}: {count}")