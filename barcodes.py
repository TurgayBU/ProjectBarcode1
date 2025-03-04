import numpy as np
import cv2
import os

from mnistmodel import MnistModel


class Barcodes:
    def __init__(self, image):
        self.image = None
        self.numbers = []
        self.model = MnistModel()  # MnistModel örneği oluştur
        self.read_image(image)

    def read_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Yüklenen dosya geçerli bir görüntü değil.")
        print(f"Görüntü '{image_path}' başarıyla yüklendi.")
        inverted_image = cv2.bitwise_not(image)
        self.image = self.make_black_white(inverted_image)

    def make_black_white(self, image):
        print("Görüntü siyah-beyaz hale getiriliyor...")
        try:
            # Global Threshold yöntemi ile daha net beyaz/siyah dönüşümü sağlanır
            _, bw_image = cv2.threshold(
                image,
                127,  # Piksel değerlerinin eşik noktası
                255,  # Eşik üstündeki piksel beyaz yapılır
                cv2.THRESH_BINARY_INV  # Siyah/Beyaz ters çevrilir (iç beyaz olması için)
            )
        except Exception as e:
            print(f"Eşikleme işlemi sırasında bir hata oluştu: {e}")
            raise ValueError("Siyah-beyaz işleme sırasında hata oluştu.")

        # Görüntüyü temizliyoruz (gereksiz gürültüleri kaldırmak için)
        return self.clean_image(bw_image)

    def clean_image(self, image):
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Geçerli bir görüntü sağlanmadı.")

        # Median blur filtresi ve ters çevirme
        cleaned_image = cv2.medianBlur(image, 3)
        processed_image = cv2.bitwise_not(cleaned_image)
        self.image = processed_image  # Görüntüyü `self.image` olarak kaydediyoruz.
        print("Görüntü başarıyla temizlenip işlendi.")
        return processed_image

    def find_boundary(self, image):
        if image is None:
            raise ValueError("Görüntü tespit edilmedi!")

        # Kontur tespiti
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Görüntü üzerinde hiçbir kontur bulunamadı.")

        image_height, image_width = image.shape[:2]
        print(f"Toplam bulunan kontur sayısı: {len(contours)}")

        # Konturları soldan sağa sıralıyoruz
        sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        self.numbers = []
        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)
            padding = 5
            x =max(0, x-padding)
            y =max(0, y-padding)
            w+=padding*4
            h+=padding*4
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Mavi çerçeve
            cv2.imwrite("debug/contours_debug.jpg", image)

            #if not (min_width <= w <= max_width) or not (min_height <= h <= max_height):
             #   print(f"Kontur atlandı: x={x}, y={y}, w={w}, h={h}")
              #  continue
            adjusted_contour = np.array(contour)
            # Konturu işle
            self.make_bound_rect(contour)

        if len(self.numbers) != 6:
            print(f"Hata: 6 kontur tespit edilemedi. Bulunan kontur sayısı: {len(self.numbers)}.")

    def make_bound_rect(self, contour):
        # Konturdan bounding box alınır
        x, y, w, h = cv2.boundingRect(contour)

        # Görüntüyü belirlenen çerçeveden kesiyoruz
        cropped_image = self.image[y:y + h, x:x + w]

        # Siyah alanları temizle (aktif kontur alanını belirle ve kes)
        bounding_box = cv2.boundingRect(cv2.findNonZero(cropped_image))  # Siyah alanı kaldır
        bx, by, bw, bh = bounding_box
        cropped_image = cropped_image[by:by + bh, bx:bx + bw]

        # Görüntünün kenarına siyah boşluk eklemek için padding tanımlıyoruz
        padding = 20  # Her kenardan 20 piksel ekle

        # Siyah boşluk eklemek
        padded_image = cv2.copyMakeBorder(
            cropped_image,  # Kesilen görüntü
            top=padding,  # Üst tarafa boşluk
            bottom=padding,  # Alt tarafa boşluk
            left=padding,  # Sol tarafa boşluk
            right=padding,  # Sağ tarafa boşluk
            borderType=cv2.BORDER_CONSTANT,  # Sabit renkli çerçeve türü
            value=[0, 0, 0]  # Siyah (RGB siyah renk: [0,0,0])
        )

        # Siyah boşluk eklenmiş işlenmiş görüntünün debug için kaydedilmesi
        debug_path = f"debug/padded_{x}_{y}.jpg"
        os.makedirs("debug", exist_ok=True)
        cv2.imwrite(debug_path, padded_image)
        print(f"Siyah boşluk eklenmiş kontur görüntüsü debug klasörüne kaydedildi: {debug_path}")

        # Görseli yeniden boyutlandır ve modelle tahmin edin
        resized_image = self.digit_resize(padded_image)
        self.numbers.append(self.predict_number(resized_image))

    def digit_resize(self, image):
        if image is None or image.size == 0:
            raise ValueError("Boş bir görüntü yeniden boyutlandırılamaz.")

        resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        normalized_image = resized_image.astype("float32") / 255.0  # Normalize et (0-1 aralığına getirilmiş)
        print(f"Yeniden boyutlandırılmış ve normalize edilmiş görüntü: {normalized_image.shape}")
        return np.expand_dims(normalized_image, axis=-1)  # HWC formatında tek kanal

    def combine_numbers(self):
        if len(self.numbers) != 6:
            print(f"Hata: Beklenen sayı adedi 6 değil. Tahmin edilenler: {self.numbers}")
            return "HATA"

        final_number = ''.join(map(str, self.numbers))
        print(f"Tahmin edilen tam sayı: {final_number}")
        return final_number

    def predict_number(self, image):
        try:
            result = self.model.predict_number(image)  # MnistModel üzerinden tahmin yap
            print(f"Model tahmini: {result}")
            return result
        except Exception as e:
            print(f"Model tahmini sırasında bir hata oluştu: {e}")
            raise ValueError("Model tahmini yapılamadı.")

    def process_and_predict(self):
        if self.image is None:
            raise ValueError("Görüntü yüklenmemiş!")
        self.find_boundary(self.image)
        return self.combine_numbers()
