import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Veri Setini Yükleme
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Veriyi Ön İşleme
# Normalize etme (0-255 arasındaki değerleri 0-1 arasına sıkıştırma)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Label'ları One-Hot Encoding'e çevirme
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Modeli Tanımlama
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 28x28 boyutundaki görüntüyü düzleştirir
    Dense(128, activation='relu'),  # 128 nöronlu gizli katman
    Dense(64, activation='relu'),   # 64 nöronlu ikinci gizli katman
    Dense(10, activation='softmax') # Çıkış katmanı (10 sınıf için softmax)
])

# Modeli Derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modeli Eğitme
epochs = 10  # Eğitim epoch sayısı
batch_size = 32  # Batch boyutu

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Modeli Test Etme
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

model.save(r'C:\Users\lenovo\Desktop\Derin Öğrenme Çalışmaları\Deep_Learning_Algorithms\mnist_model_deep_learning.h5')

# Örnek Tahmin
import numpy as np
import matplotlib.pyplot as plt

# Rastgele bir test görüntüsü seçme
random_index = np.random.randint(0, len(x_test))
random_image = x_test[random_index]
prediction = model.predict(random_image.reshape(1, 28, 28))
predicted_label = np.argmax(prediction)

plt.imshow(random_image, cmap='gray')
plt.title(f"Tahmin: {predicted_label}")
plt.show()