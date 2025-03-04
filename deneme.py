import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# MNIST veri kümesini yükle
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Eğitim ve test verisi ayrımı

# Giriş verilerini normalleştir
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Model oluştur
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # Düzleştirme
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))  # Birinci gizli katman
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))  # İkinci gizli katman
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))  # Çıkış katmanı

# Modeli derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğit
model.fit(x_train, y_train, epochs=3)  # Epoch değerini isteğe bağlı artırabilirsiniz

# Modeli test et
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test doğruluğu: {accuracy}")
print(f"Test kaybı: {loss}")


# Test edilmek üzere görsel yükleme ve işleme fonksiyonu
def load_and_process_image(image_path):
    """
    Görseli yükler, siyah-beyaz hale getirir, boyutlandırır ve model için uygun hale getirir.
    """
    # Görseli yükle (gri tonlama olarak)
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

    # Görseli invert et (siyah-beyaz ters çevir)
    img = cv.bitwise_not(img)

    # Görseli model için uygun boyuta getirme
    img = cv.resize(img, (28, 28))  # MNIST'in beklediği 28x28 boyutu
    img = img.astype("float32") / 255.0  # Normalize et (0-1 aralığına getir)
    img = np.expand_dims(img, axis=-1)  # Kanal boyutu ekle (28x28x1)
    img = np.expand_dims(img, axis=0)  # Batch boyutu ekle (1x28x28x1)
    img = im
    return img


# Seçilen görüntüyü tahmin et ve sonucu göster
def predict_image(model, image_path):
    """
    İşlenmiş görseli kullanarak tahmin yapar ve sonucu terminalde gösterir.
    """
    processed_image = load_and_process_image(image_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)

    # Tahmin sonuçlarını görsel olarak göster
    print("-----------------")
    print(f"Tahmin edilen değer: {predicted_class}")
    print("-----------------")

    # Görseli yükle (işlenmemiş halini göstermek için)
    original_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    plt.imshow(original_image, cmap=plt.cm.binary)  # Siyah-beyaz göster
    plt.title(f"Tahmin: {predicted_class}")
    plt.axis('off')
    plt.show()


# Test görüntüsüyle tahmin yap
image_path = 'uploads/9.jpeg'  # Görsel yolunu doğru verdiğinizden emin olun
try:
    predict_image(model, image_path)
except Exception as e:
    print(f"Hata oluştu: {e}")
