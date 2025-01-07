# depresyon durumunu etkileyen faktörleri analiz etmek, görselleştirmek ve lojistik regresyon modeliyle depresyonda olma durumunu tahmin etmek için bir veri seti kullandık
# 0: Depresyonda olmama durumu
# 1: Depresyonda olma durumu

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("veri_seti.csv")

print(df.head())
print(df.info())

print(df.isnull().sum())
df.fillna(df.mean(), inplace=True) # Eksik verileri örneğin, ortalama ile doldurma

sns.countplot(x='Gender', hue='Depression', data=df)
plt.title("Cinsiyet ve Depresyon Durumu")
plt.show()

sns.boxplot(x='Depression', y='Age', data=df)
plt.title("Yaş ve Depresyon Durumu")
plt.show()

sns.violinplot(x='Depression', y='Academic Pressure', data=df)
plt.title("Akademik Baskı ve Depresyon Durumu")
plt.show()

sns.histplot(df[df['Depression'] == 1]['Sleep Duration'], kde=True, color='red', label='Depressed')
sns.histplot(df[df['Depression'] == 0]['Sleep Duration'], kde=True, color='green', label='Not Depressed')
plt.legend()
plt.title("Uyku Süresi ve Depresyon Durumu")
plt.show()

corr_columns = ['Age', 'Academic Pressure', 'Study Satisfaction', 'Sleep Duration', 
                'Dietary Habits', 'Study Hours', 'Financial Stress', 'Family History of Mental Illness']
corr_matrix = df[corr_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korelasyon Matrisi")
plt.show()

X = df[['Age', 'Academic Pressure', 'Study Satisfaction', 'Sleep Duration', 
        'Dietary Habits', 'Study Hours', 'Financial Stress', 'Family History of Mental Illness']]
y = df['Depression']  # Hedef değişken (Depresyon Durumu)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))