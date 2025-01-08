# TensorFlow, yapay zeka modelleri oluşturmak, eğitmek ve dağıtmak için kullanılan açık kaynaklı bir makine öğrenimi kütüphanesidir.
# Kod, bir sinir ağı kullanarak AND kapısının giriş-çıkış ilişkisini öğrenmek ve bu ilişkiye göre tahmin yapmak için TensorFlow ile bir model oluşturur, eğitir ve değerlendirir.
# Sinir ağı, insan beynindeki sinapsların çalışma prensibini taklit ederek veriler arasındaki ilişkileri öğrenen ve tahminler yapan bir yapay zeka modelidir.

import tensorflow as tf
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Girdi
Y = np.array([0, 0, 0, 1])  # Çıktı

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_dim=2, activation='sigmoid')  # Perceptron
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, Y, epochs=1000, verbose=0)

predictions = model.predict(X)
print("Gerçek Sonuçlar: ", Y)
print("Tahminler: ", (predictions > 0.5).astype(int))  # 0.5'ten büyükse 1, değilse 0

loss, accuracy = model.evaluate(X, Y)
print(f"Modelin doğruluk oranı: {accuracy * 100:.2f}%")