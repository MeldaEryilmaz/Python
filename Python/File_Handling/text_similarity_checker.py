import os
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):

    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator).lower()

def remove_support_words(text, stopwords):

    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

def check_text_similarity(text1, text2):

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

stopwords_file = os.path.join("File_Handling", "stop_words.py")
try:
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()
except FileNotFoundError:
    print(f"Durdurma sözcükleri dosyası bulunamadı: {stopwords_file}")
    stopwords = []

michelle_file = os.path.join("File_Handling", "michelle_obama_speech.txt")
melina_file = os.path.join("File_Handling", "melina_trump_speech.txt")

try:
    with open(michelle_file, 'r', encoding='utf-8') as file:
        michelle_text = file.read()
    with open(melina_file, 'r', encoding='utf-8') as file:
        melina_text = file.read()

    michelle_cleaned = remove_support_words(clean_text(michelle_text), stopwords)
    melina_cleaned = remove_support_words(clean_text(melina_text), stopwords)

    similarity_score = check_text_similarity(michelle_cleaned, melina_cleaned)
    print(f"Michelle'in ve Melina'nın konuşmaları arasındaki benzerlik: {similarity_score:.2f}")

except FileNotFoundError as e:
    print(f"Dosya bulunamadı: {e}")