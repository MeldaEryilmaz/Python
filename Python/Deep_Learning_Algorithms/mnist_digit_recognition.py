import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

# MNIST modelini eğit
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# Çizim arayüzü
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rakam Çizimi")
        
        self.canvas = tk.Canvas(root, width=280, height=280, bg="black")  # Siyah arka plan
        self.canvas.pack()
        
        self.image = Image.new("L", (28, 28), color="black")  # Siyah arka plan
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.button_frame = tk.Frame(root)
        self.button_frame.pack()
        
        self.predict_button = tk.Button(self.button_frame, text="Tahmin Et", command=self.predict)
        self.predict_button.pack(side="left")
        
        self.clear_button = tk.Button(self.button_frame, text="Temizle", command=self.clear)
        self.clear_button.pack(side="right")

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="white", outline="white")
        self.draw.ellipse([x//10-1, y//10-1, x//10+1, y//10+1], fill="white")
        
    def predict(self):
        image_array = np.array(self.image.resize((28, 28))) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        prediction = model.predict(image_array)
        predicted_label = np.argmax(prediction)
        
        result_label = tk.Label(self.root, text=f"Tahmin: {predicted_label}", font=("Helvetica", 16))
        result_label.pack()
        
    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), color="black")
        self.draw = ImageDraw.Draw(self.image)

# GUI'yi çalıştır
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
