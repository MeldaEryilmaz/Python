import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw

# MNIST verisi yükleniyor
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Veriyi normalleştiriyoruz
x_train, x_test = x_train / 255.0, x_test / 255.0

# Modeli tanımlıyoruz
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Modeli derliyoruz
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitiyoruz (önceden eğitilmiş veriyi kullanarak)
model.fit(x_train, y_train, epochs=5)

# Yeni çizimi almak için tkinter arayüzü
class DrawApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Rakam Tahmin Uygulaması")
        self.master.geometry("400x450")  # Pencere boyutunu artırdık
        
        self.canvas = tk.Canvas(self.master, width=280, height=280, bg='white')
        self.canvas.pack(pady=20)
        
        self.button_clear = tk.Button(self.master, text="Temizle", width=15, command=self.clear_canvas)
        self.button_clear.pack(pady=5)
        
        self.button_predict = tk.Button(self.master, text="Tahmin Et", width=15, command=self.predict_digit)
        self.button_predict.pack(pady=5)
        
        # Tahmin sonucunu göstermek için label
        self.result_label = tk.Label(self.master, text="Tahmin Sonucu: ", font=("Arial", 14))
        self.result_label.pack(pady=10)

        self.lastx, self.lasty = None, None
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Canvas'tan çizim alabilmek için PIL imaj nesnesi
        self.image = Image.new("L", (280, 280), 255)  # "L" grayscale mode
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        if self.lastx and self.lasty:
            self.canvas.create_line(self.lastx, self.lasty, x, y, width=8, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.lastx, self.lasty, x, y], fill=0, width=8)
        self.lastx, self.lasty = x, y
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)  # Canvas'ı temizle ve yeni bir boş resim oluştur
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Tahmin Sonucu: ")  # Sonucu temizle

    def predict_digit(self):
        # Canvas içeriğini img olarak al
        img = self.image.resize((28, 28))  # 28x28 boyutuna küçült
        img = np.array(img)  # NumPy array'e dönüştür
        
        # Normalizasyon (0-255 arası -> 0-1 arası)
        img = img.astype('float32') / 255.0
        
        # Siyah beyaz ters çevirme (çünkü tkinter beyaz arka planda siyah çizer)
        img = 255 - img  # Siyah beyaz ters çevirme
        
        # Modelin giriş formatına uygun şekle getirme (28, 28, 1)
        img = img.reshape(1, 28, 28, 1)
        
        # Modelin tahmin yapabilmesi için, son aşamada modelin tahminini al
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)
        
        # Sonucu GUI üzerinde göster
        self.result_label.config(text=f"Tahmin Sonucu: {predicted_digit}")

# tkinter penceresini başlat
root = tk.Tk()
app = DrawApp(root)
root.mainloop()
