import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

path_train = os.path.join("C:/Users/lenovo/Desktop/Derin Öğrenme Çalışmaları/MNIST", "train.csv")
path_test = os.path.join("C:/Users/lenovo/Desktop/Derin Öğrenme Çalışmaları/MNIST", "test.csv")
train = pd.read_csv(path_train)
test = pd.read_csv(path_test)

print("train shape: ", train.shape)
print("test shape: ", test.shape)
train.head(3)

labels = train.iloc[:, :1]
train_data = train.iloc[:, 1:]

train_data = train_data / 255.0

labels = to_categorical(labels, 10)

X_train, X_val, y_train, y_val = train_test_split(train_data, labels, test_size=0.2, random_state=42)

# Veriyi yeniden şekillendir (28x28x1)
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_val = X_val.values.reshape(-1, 28, 28, 1)
X_test = test.values.reshape(-1, 28, 28, 1)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=200, verbose=2)

test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)
print("\nTest accuracy: {:.2f}%".format(test_acc * 100))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.tight_layout()  # Grafiklerin düzgün görünmesi için
plt.show()

# Sonuçları tahmin et
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)

# Sonuçları Kaggle formatına göre kaydet
submission_path = os.path.join("C:/Users/lenovo/Desktop/Derin Öğrenme Çalışmaları/MNIST", "submission.csv")
submission = pd.DataFrame({'ImageId': np.arange(1, len(predictions) + 1), 'Label': predictions})
submission.to_csv(submission_path, index=False)
print(f"Prediction saved to '{submission_path}'")