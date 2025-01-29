import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Veri setini yükleyip işleyelim
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0  
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)  
y_test = to_categorical(y_test, 10)

# Modeli tanımlayalım
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# SGD ile eğitilecek model
model_sgd = create_model()
model_sgd.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Adam ile eğitilecek model
model_adam = create_model()
model_adam.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# Modelleri eğitelim
history_sgd = model_sgd.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=1)
history_adam = model_adam.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=1)

plt.figure(figsize=(12, 6))

# Kayıp (Loss) grafiği
plt.subplot(1, 2, 1)
plt.plot(history_sgd.history['loss'], label='SGD Loss')
plt.plot(history_adam.history['loss'], label='Adam Loss')
plt.title('Eğitim Kayıpları')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Doğruluk (Accuracy) grafiği
plt.subplot(1, 2, 2)
plt.plot(history_sgd.history['accuracy'], label='SGD Accuracy')
plt.plot(history_adam.history['accuracy'], label='Adam Accuracy')
plt.title('Eğitim Doğrulukları')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Test seti sonuçlarını karşılaştıralım
loss_sgd, acc_sgd = model_sgd.evaluate(x_test, y_test, verbose=0)
loss_adam, acc_adam = model_adam.evaluate(x_test, y_test, verbose=0)

print("SGD Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(loss_sgd, acc_sgd))
print("Adam Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(loss_adam, acc_adam))