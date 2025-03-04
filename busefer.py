import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. Veri Setini Yükleme ve Hazırlama
# MNIST veri setini yükle
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizasyon ve format: Piksel değerlerini 0-255 yerine 0-1'e ölçekle
x_train = x_train / 255.0
x_test = x_test / 255.0

# Renk kanalı ekle: (28x28x1) boyutuna dönüştür
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Etiketleri one-hot encoding'e dönüştür
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# 2. Modelin Tanımlanması
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Overfitting'i önlemek için Dropout kullanıldı
    Dense(10, activation='softmax')  # Çıkış 10 sınıflı olacak (0-9)
])

# 3. Modelin Derlenmesi
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model özeti yazdır
model.summary()

# 4. Erken Durdurma Callback'i
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 5. Modeli Eğit
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=20,  # Kaç epok boyunca eğitim yapılacağı
    batch_size=64,  # Toplu veri boyutu
    callbacks=[early_stopping],  # Erken durdurmayı etkinleştir
    verbose=1
)

# 6. Performans Analizi
# Eğitim ve doğrulama başarımı
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Doğruluğu: {test_acc:.2f}")

# Eğitim sürecinde doğruluk ve kayıp değerlerini görselleştir
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluk')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp')
plt.legend()

plt.show()

# 7. Eğitilen Modeli Kaydet
try:
    model.save("mnist_model_mark1.keras")
    print("Model başarıyla kaydedildi.")
except Exception as e:
    print("Model kaydedilemedi! Hata:")
    print(e)

print("\nModel başarıyla eğitildi ve 'mnist_model.h5' dosyası olarak kaydedildi!")
