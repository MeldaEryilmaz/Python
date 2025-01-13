import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import cv2
import os

# Modeli yükle
model = tf.keras.models.load_model('C:/Users/lenovo/Desktop/Derin Öğrenme Çalışmaları/MNIST/digit_recognition_model.h5')

# Yeni çizimi almak için tkinter arayüzü
class DrawApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Rakam Tahmin Uygulaması")
        self.master.geometry("400x400")
        
        self.canvas = tk.Canvas(self.master, width=280, height=280, bg='white')
        self.canvas.pack(pady=20)
        
        self.button_clear = tk.Button(self.master, text="Temizle", width=15, command=self.clear_canvas)
        self.button_clear.pack(pady=5)
        
        self.button_predict = tk.Button(self.master, text="Tahmin Et", width=15, command=self.predict_digit)
        self.button_predict.pack(pady=5)
        
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
        
        # Sonucu göster
        messagebox.showinfo("Tahmin Sonucu", f"Tahmin edilen rakam: {predicted_digit}")

# tkinter penceresini başlat
root = tk.Tk()
app = DrawApp(root)
root.mainloop()
