import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# 1. Veri Yükleme ve Model Yükleme
mnist = fetch_openml("mnist_784")
X = mnist.data / 255.0  # Normalize et (0-255 arası, bunu 0-1 arası yapıyoruz)
y = mnist.target.astype(int)

# Modeli yükleyin (önceden eğitilmiş modelin yolunu belirtin)
model = load_model('mnist_model.keras')  # Modelinizi yola göre değiştirin

# 2. Test Verisini Ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 9 Sayılarının İndekslerini Bul
nine_indices = [i for i, label in enumerate(y_test) if label == 9]

# 4. 9 Sayılarının Görsellerini Seç
nine_images = X_test.iloc[nine_indices]

# 5. Görselleri (28, 28, 1) Şeklinde Yeniden Şekillendir
nine_images_reshaped = nine_images.values.reshape(-1, 28, 28, 1)

# 6. Modeli Kullanarak Tahmin Yap
predictions = model.predict(nine_images_reshaped)

# 7. Görselleri 5x5 Grid Şeklinde Göster
plt.figure(figsize=(10, 10))

for i in range(25):  # İlk 25 9'u görselleştir
    plt.subplot(5, 5, i + 1)
    plt.imshow(nine_images_reshaped[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.title(f"Pred: {np.argmax(predictions[i])}")  # Tahmin edilen sonuç

plt.show()
