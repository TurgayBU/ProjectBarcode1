import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Model dosyasını yükleyin
model_path = 'hadi_ins.keras'  # Modelin doğru formatta kaydedildiğini kontrol et
model = load_model(model_path)

# Test için bir görüntü ekleyin
image_path = 'uploads/9.jpeg'  # Test etmek için bir görselin yolu


# Görüntüyü işleme ve görsel çıktısı
def preprocess_image(image_path):
    """
    Görseli işleyen bir fonksiyon: gürültü azaltma, eşiğe alma, ters çevirme ve boyutlandırma işlemleri içerir.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Görseli siyah-beyaz oku
    if image is None:
        raise FileNotFoundError(f'Görsel {image_path} bulunamadı!')

    # Gürültü azaltma ve kontrast artırma
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Gürültüyü azalt
    image = cv2.equalizeHist(image)  # Kontrastı artır

    # Siyah-beyaz eşikleme
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Görüntüyü ters çevir (isteğe bağlı)
    # Eğer sonuçlar kötü ise, bu işlemi kaldırabilirsiniz
    image = 255 - image  # Ters çevirme işlemi

    # Görüntüyü 28x28 boyutuna getir ve yeniden ölçekle
    image = cv2.resize(image, (28, 28))  # MNIST boyutuna getir

    # Görselin nasıl göründüğünü kontrol et (ön işleme sonrası)
    plt.imshow(image, cmap="gray")
    plt.title("İşlenen Görüntü")
    plt.axis('off')
    plt.show()

    # Normalizasyon (model girişi için veriyi 0-1 aralığına getir)
    image = image.astype("float32") / 255.0  # 0-255 aralığındaki değerleri normalize et
    image = np.expand_dims(image, axis=-1)  # Kanal boyutunu ekle (28x28x1)
    image = np.expand_dims(image, axis=0)  # Batch boyutunu ekle (1x28x28x1)
    return image


# Model tahmini ve sonuç görselleştirme
def predict_and_display(model, preprocessed_image, original_image_path):
    """
    İşlenmiş görseli kullanarak model tahmini yapar ve sonucu gösterir.
    """
    predictions = model.predict(preprocessed_image)
    predicted_number = np.argmax(predictions)  # En yüksek olasılıklı tahmin edilen sınıf
    confidence = np.max(predictions)  # Tahmin edilen sınıfın güven skoru

    # Modelin tüm olasılıklarını yazdır
    print("Tahmin Olasılıkları:")
    for i, prob in enumerate(predictions[0]):
        print(f"{i}: {prob:.5f}")

    # Orijinal görseli de göster
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(original_image, cmap="gray")
    plt.title(f"Gerçek Görsel - Tahmin Edilen: {predicted_number}")
    plt.axis('off')
    plt.show()

    # Sonuçları yazdır
    print(f"\nTahmin Edilen Sayı: {predicted_number}")
    print(f"Güven Skoru: {confidence:.5f}")


# Ana çalışma fonksiyonu
def main():
    try:
        # Görseli işleyin
        processed_image = preprocess_image(image_path)

        # Modelden tahmin alın ve sonucu göster
        predict_and_display(model, processed_image, image_path)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Bir hata oluştu: {e}")


if __name__ == "__main__":
    main()
