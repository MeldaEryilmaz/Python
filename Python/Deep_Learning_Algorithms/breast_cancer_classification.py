# Kod, modelin test verisi (Meme Kanseri Veri Kümesi) üzerindeki tahminlerini alır, bunları ikili sınıflara (iyi huylu veya kötü huylu) dönüştürür ve sınıflandırma sonuçlarını elde eder.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/lenovo/Desktop/Derin Öğrenme Çalışmaları/Deep_Learning_Algorithms/breast_cancer_dataset.csv")

print(data.head())
print(data.info())

X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(32, activation='relu', input_dim=X_train.shape[1]),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu', color='blue')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu', color='orange')
plt.legend()
plt.title('Model Doğruluğu')
plt.xlabel('Epochs')
plt.ylabel('Doğruluk')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı', color='blue')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı', color='orange')
plt.legend()
plt.title('Model Kaybı')
plt.xlabel('Epochs')
plt.ylabel('Kayıp')

plt.tight_layout()
plt.show()

y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Tahmin Sonuçları:", y_pred.flatten())
# Eğer tahmin 0.5'ten büyükse, model örneği "kötü huylu" (1) olarak sınıflandırır.
# Eğer tahmin 0.5'ten küçükse, model örneği "iyi huylu" (0) olarak sınıflandırır.