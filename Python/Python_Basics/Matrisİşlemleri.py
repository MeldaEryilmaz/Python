import numpy as np

"""
# manuel matris oluşturma
A = np.array([[1,2,3], [4,5,6], [7,8,9]])
print("Matris A:\n", A) 

# rastgele matris oluşturma
B = np.random.randint(1, 10, size=(3,3))
print("rastgele matris B:\n", B)

# birim matris oluşturma
I = np.eye(3)
print("Birim matris I: \n", I)

# sıfır ve bir matris
zeros = np.zeros((3,3))
ones = np.ones((2,4))
"""

def matris_olustur():
    rows = int(input("Matrisin satır sayısını girin: "))
    cols = int(input("Matrisin sütun sayısını girin: "))
    print(f"{rows}x{cols} boyutunda matris elemanlarını sırayla girin:")
    matris = []
    for i in range(rows):
        row = list(map(float, input(f"{i+1}. satır elemanlarını girin (boşluklarla ayırın): ").split()))
        matris.append(row)
    return np.array(matris)

def matris_islemleri():
    """Matris işlemleri için bir menü sunar."""
    print("Matris A'yı oluşturun:")
    A = matris_olustur()
    print("\nMatris B'yi oluşturun:")
    B = matris_olustur()
    
    while True:
        print("\nMatris İşlemleri Menüsü:")
        print("1. Matris Toplama (A + B)")
        print("2. Matris Çıkarma (A - B)")
        print("3. Matris Çarpma (A * B - Dot Product)")
        print("4. Matrisin Transpozu (A ve B)") # satırlar sütun, sutünlar satır oluyor
        print("5. Matrisin Tersini Al (A ve B)")
        print("6. Determinant Hesapla (A ve B)")
        print("7. Çıkış")
        
        secim = input("\nBir işlem seçin (1-7): ")
        
        if secim == "1":
            try:
                print("\nA + B Sonucu:\n", A + B)
            except ValueError:
                print("Hata: Matris boyutları toplama için uygun değil!")
        
        elif secim == "2":
            try:
                print("\nA - B Sonucu:\n", A - B)
            except ValueError:
                print("Hata: Matris boyutları çıkarma için uygun değil!")
        
        elif secim == "3":
            try:
                print("\nA * B (Dot Product) Sonucu:\n", np.dot(A, B))
            except ValueError:
                print("Hata: Matris boyutları çarpma için uygun değil!")
        
        elif secim == "4":
            print("\nA Matrisinin Transpozu:\n", A.T)
            print("B Matrisinin Transpozu:\n", B.T)
        
        elif secim == "5":
            try:
                print("\nA Matrisinin Tersi:\n", np.linalg.inv(A))
            except np.linalg.LinAlgError:
                print("Hata: A matrisi kare değil veya terslenemez!")
            try:
                print("\nB Matrisinin Tersi:\n", np.linalg.inv(B))
            except np.linalg.LinAlgError:
                print("Hata: B matrisi kare değil veya terslenemez!")
        
        elif secim == "6":
            try:
                print("\nA Matrisinin Determinantı:", np.linalg.det(A))
            except np.linalg.LinAlgError:
                print("Hata: A matrisinin determinantı hesaplanamıyor!")
            try:
                print("B Matrisinin Determinantı:", np.linalg.det(B))
            except np.linalg.LinAlgError:
                print("Hata: B matrisinin determinantı hesaplanamıyor!")
        
        elif secim == "7":
            print("Programdan çıkılıyor...")
            break
        
        else:
            print("Geçersiz seçim! Lütfen 1-7 arasında bir sayı girin.")

if __name__ == "__main__":
    matris_islemleri()