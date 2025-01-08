# Görselleştirdiğimizde eksenler neyi ifade ederler?
# X Ekseninde: Veri noktalarının sıralanması veya veri index'i (sadece görsel amacıyla).
# Y Ekseninde: Weekly_Sales (Haftalık Satış) değerleri.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

file_path = "walmart_sales.csv" 
df = pd.read_csv(file_path)

print("Veri seti başlıkları:")
print(df.head())

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')  
    df['Date'] = df['Date'].map(pd.Timestamp.toordinal)  

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Weekly_Sales', axis=1)  # 'Weekly_Sales' bağımlı değişken
y = df['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color="blue", label="Gerçek Değerler")
plt.scatter(range(len(y_pred)), y_pred, color="red", label="Tahmin Edilen Değerler")
plt.xlabel("Veri Noktaları")
plt.ylabel("Weekly Sales")
plt.title("Gerçek Değerler vs Tahmin Edilen Değerler")
plt.legend()
plt.show()

results = pd.DataFrame({"Gerçek Değerler": y_test.values, "Tahmin Edilen Değerler": y_pred})
print("\nTahmin Sonuçları:")
print(results)