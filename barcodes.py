import numpy as np
import cv2
import os

from mnistmodel import MnistModel


class Barcodes:
    def __init__(self, image):
        self.image = None
        self.numbers = []
        self.model = MnistModel()  # MnistModel örneği
        self.read_image(image)

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image format not suitable for MNIST. Try again with a different image format.")
        print(f"Image '{image_path}' successfully loaded..")

        # Sadece mavi tonlarına yönelik kontrast artırımı
        enhanced_image = self.enhance_blue_tones(image)

        # Renkli görüntüyü gri tonlamaya dönüştür (diğer işlemler için)
        grayscale_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

        inverted_image = grayscale_image
        self.image = self.make_black_white(inverted_image)
        #self.image = self.apply_vertical_crop_filter(self.image, height_threshold=700, crop_percent=0.70)

    def make_black_white(self, image):
        print("Converting image to black and white with guaranteed black background...")
        try:
            # Gri tonlamalı hale getir (eğer değilse)
            if len(image.shape) > 2:
                print("Image is not grayscale. Converting to grayscale...")
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            # Gauss bulanıklaştırma (gürültüyü azaltmak için)
            blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
            _,bw_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)
            # Adaptif threshold ile görüntüyü siyah-beyaza çevir
            bw_image = cv2.adaptiveThreshold(
                blurred_image,
                255,  # Maksimum beyaz değeri
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Komşu alanın ortalamasına dayalı daha iyi gaussla
                cv2.THRESH_BINARY_INV,  # Arka plan beyaz, öndeki desenler siyah
                13,  # Pencere boyutu
                5  # Sabit değer
            )

            print("Adaptive thresholding applied.")

            # MORFOLOJİK BOŞLUK DOLDURMA EKLENDİ
            bw_image = self.fill_close_gaps(bw_image, kernel_size=5, iterations=2)  # Küçük boşlukları doldur
            print("Small gaps between white pixels filled.")
            bw_image = self.remove_lines_safely(bw_image, min_line_length=100, max_line_thickness=8)
            bw_image = self.remove_lines_with_safe_margin(bw_image, margin=15)
            # Arka planın siyah olması için görüntüyü ters çevir
            inverted_bw_image = cv2.bitwise_not(bw_image)
            print("Inverted the image to make background black.")

        except Exception as e:
            print(f"An error occurred during black and white conversion: {e}")
            raise ValueError("Error during adaptive thresholding or inversion.")

        # Gürültüyü temizleme ve elde edilen görüntüyü döndürme
        cleaned_image = self.clean_image(inverted_bw_image)
        print("Cleaned the image to remove noise.")

        return cleaned_image

    def clean_image(self, image):
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Not provided image or not a numpy array.")

        # Median blur filtresi ve ters çevirme
        cleaned_image = cv2.medianBlur(image, 3)
        processed_image = cv2.bitwise_not(cleaned_image)
        self.image = processed_image  # Görüntüyü `self.image` olarak kaydediyok
        print("Image cleaned and inverted..")
        return processed_image

    def find_boundary(self, image):
        if image is None:
            raise ValueError("Image not loaded!")

        # Kontur tespiti
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours found in the image. Please verify the image.")

        print(f"Total contours detected: {len(contours)}")

        # Alanı küçük olan (boşlukların oluşturduğu küçük objeler) konturları temizlemek
        merged_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Sadece minimum 50 piksel alanı olan konturlar
                merged_contours.append(contour)

        print(f"Contours after filtering by area: {len(merged_contours)}")

        # Konturları alanlarına göre sırala
        sorted_contours = sorted(merged_contours, key=lambda c: cv2.contourArea(c), reverse=True)

        # Sadece en büyük 6 konturu al
        main_contours = sorted_contours[:6] if len(sorted_contours) >= 6 else sorted_contours

        # Konturları soldan sağa sıralayın
        sorted_main_contours = sorted(main_contours, key=lambda c: cv2.boundingRect(c)[0])

        # Debug için konturlarla görüntü oluştur:
        debug_contours_image = image.copy()
        for c in sorted_main_contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(debug_contours_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        os.makedirs("debug", exist_ok=True)
        cv2.imwrite("debug/contours_debug.jpg", debug_contours_image)

        # İşlenmiş konturları sayılara ayır
        self.numbers = []
        for contour in sorted_main_contours:
            self.make_bound_rect(contour)

    def make_bound_rect(self, contour):
        # Konturdan bounding box (dikdörtgen sınır) al
        x, y, w, h = cv2.boundingRect(contour)

        # Görüntüyü bounding box içerisinde kes (kesit net olacak)
        cropped_image = self.image[y:y + h, x:x + w]

        # Siyah arka plan üzerinde padding ekle (beyaz alanlar eklenmez)
        padding = 20  # İhtiyaç kadar padding değeri
        padded_image = cv2.copyMakeBorder(
            cropped_image,
            top=padding, bottom=padding, left=padding, right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Siyah padding ekler
        )

        # Flood Fill kullanarak görüntünün iç siyah kısımlarını koru
        protected_image = self.preserve_inner_black_regions(padded_image)

        # Görüntüyü yeniden boyutlandır ve MNIST modeli için tahmin et
        resized_image = self.digit_resize(protected_image)

        # Tahmin edilen sayıyı alın
        predicted_number = self.predict_number(resized_image)
        self.numbers.append(predicted_number)

        # Ayrılan sayıyı debug klasörüne kaydedin
        os.makedirs("debug", exist_ok=True)  # 'debug' klasörünü oluştur
        debug_path = f"debug/number_{len(self.numbers)}.jpg"  # Sayılar sıralı olarak kaydedilecek
        cv2.imwrite(debug_path, cropped_image)
        print(f"Number debug image saved: {debug_path}")

    def digit_resize(self, image):
        if image is None or image.size == 0:
            raise ValueError("Empty image not working.")

        resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        normalized_image = resized_image.astype("float32") / 255.0  # Normalize et (0-1 aralığına getirilme)
        print(f"Reshaped and normalized image: {normalized_image.shape}")
        return np.expand_dims(normalized_image, axis=-1)  # HWC formatında tek kanal

    def combine_numbers(self):
        if len(self.numbers) != 6:
            print(f"Error: Waited numbers not 6 digit. Predicteds: {self.numbers}")
            return "ERROR"

        final_number = ''.join(map(str, self.numbers))
        print(f"Predicted integer: {final_number}")
        return final_number

    def predict_number(self, image):
        try:
            result = self.model.predict_number(image)  # MnistModel üzerinden tahmin yapma
            print(f"Model prediction: {result}")
            return result
        except Exception as e:
            print(f"Error has been occured when predicted: {e}")
            raise ValueError("Model prediction error has been occured. Please try again later or contact the developer for help. Thank you..")

    def process_and_predict(self):
        if self.image is None:
            raise ValueError("Image not loaded. Please load an image first and try again. Thank you..!")
        self.find_boundary(self.image)
        return self.combine_numbers()

    def apply_vertical_crop_filter(self, image, height_threshold=500, crop_percent=0.70):
        """
        Orta noktadan yukarı ve aşağı belirli bir oranı (örn. %70) korur, diğer tüm alanları siyah yapar.
        Yalnızca yükseklik belirtilen eşikten büyükse uygulanır.
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Image not provided or not valid numpy array.")

        height, width = image.shape[:2]

        # Yalnızca belirli yüksekliğin üzerindeki görseller için uygula
        if height < height_threshold:
            print(f"Image height {height} is smaller than threshold {height_threshold}, skipping crop filter.")
            return image

        print(f"Applying vertical crop filter: height={height}, width={width}, crop_percent={crop_percent}")

        # Korunacak bölgenin toplam yüksekliği
        keep_height = int(height * crop_percent)
        keep_margin = (height - keep_height) // 2

        keep_top = keep_margin
        keep_bottom = keep_height-100

        # Maske oluştur
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[keep_top:keep_bottom, :] = 255  # Sadece orta kısmı koruyoruz

        # Görüntünün sadece maskeye denk gelen kısmını bırak, geri kalanı siyah yap
        filtered = cv2.bitwise_and(image, mask)

        print(f"Vertical crop filter applied successfully: top={keep_top}, bottom={keep_bottom}")
        return filtered

    def remove_inner_black(self, image):
        """
        Görüntüde kapalı siyah bölgeler tespit edilir ve beyaza dönüştürülür.
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Image not provided or not a valid numpy array.")

        # Flood Fill işlemi için işlem görecek görüntünün kopyasını al
        flood_filled = image.copy()

        # Flood fill için bir maske oluştur (Maske boyutu, flood_filled'dan 2 piksel büyük olmalı)
        h, w = flood_filled.shape[:2]
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)  # Mask boyutları: flood_filled + 2 piksel

        # Flood Fill uygula (0,0 noktası siyah olan yerden başlayarak beyaz doldurur)
        seed_point = (0, 0)  # Başlangıç noktası (sol üst köşe)
        new_value = 255  # Siyah olan yerleri beyaza çevirmek için kullanılacak değer
        cv2.floodFill(flood_filled, mask, seed_point, new_value)

        # Flood fill sonrası beyaz olan bölgelerin tersini al
        flood_filled_inverse = cv2.bitwise_not(flood_filled)

        # Orijinal görüntü ve beyaz dolgu birleşimi (temiz görüntü)
        result = cv2.bitwise_or(image, flood_filled_inverse)

        print("Removed inner black areas successfully.")
        return result

    def enhance_blue_tones(self, image):
        if image is None or len(image.shape) != 3:  # Eğer görüntü renkli değilse hata
            raise ValueError("Image must be a color (BGR) image.")

        # BGR kanalları ayır
        b_channel, g_channel, r_channel = cv2.split(image)

        # Mavi kanal üzerinde CLAHE veya histogram eşitleme uygulayın
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHE nesnesi
        enhanced_blue = clahe.apply(b_channel)  # Sadece mavi kanalı iyileştiriyoruz

        # Yalnızca mavi tonlarını işlem yapmak için kümeleme yap
        mask = cv2.inRange(image, np.array([100, 0, 0]), np.array([255, 100, 100]))  # Mavi aralığı
        blue_only = cv2.bitwise_and(enhanced_blue, enhanced_blue, mask=mask)

        # Tüm kanalları birleştir
        merged_image = cv2.merge((blue_only, g_channel, r_channel))

        print("Enhanced blue tones successfully.")
        return merged_image

    def preserve_inner_black_regions(self, image):
        """
        Görüntüdeki siyah alanların (örnek: 9'un ortasındaki boş alan) korunmasını sağlar.
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image.")

        # Görüntü kopyası al
        protected_image = image.copy()

        # Flood fill uygulama
        h, w = protected_image.shape[:2]
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)  # Maske flood fill için

        # Flood Fill işlemini uygula
        seed_point = (0, 0)  # Sol üst köşeden başla
        cv2.floodFill(protected_image, mask, seed_point, 255)

        # Flood fill sonrası ters çevir (kenar beyaz, içerik siyah olacak)
        flood_filled_inverse = cv2.bitwise_not(protected_image)

        # Orijinal görüntüyü korumak için birleştir
        result = cv2.bitwise_or(image, flood_filled_inverse)

        print("Inner black regions preserved successfully.")
        return result

    def fill_close_gaps(self, image, kernel_size=3, iterations=2):
        """
        Beyaz pikseller arasındaki küçük boşlukları doldurur.
        :param image: Siyah-beyaz görüntü
        :param kernel_size: Yapısal eleman boyutu
        :param iterations: Iterasyon sayısı
        :return: Doldurulmuş görüntü
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))  # Daha yuvarlak bağlama için
        dilated_image = cv2.dilate(image, kernel, iterations=iterations)
        return dilated_image

    def remove_lines_safely(self, image, min_line_length=50, max_line_thickness=10):
        """
        Görüntüdeki çizgiyi sayılardan ayrı tespit edip kaldırır.
        :param image: Siyah-beyaz görüntü
        :param min_line_length: Minimum tespit edilecek çizgi uzunluğu
        :param max_line_thickness: Maksimum çizgi kalınlığı (kalın çizgileri hariç tutar)
        :return: Çizgileri temizlenmiş görüntü
        """
        # Görüntünün konturlarını bulun
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Çizgileri belirlemek için boş bir maske oluştur
        mask = np.zeros_like(image, dtype=np.uint8)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            # Çizgileri tespit etmek için uzun ve ince konturları hedefleyin
            if aspect_ratio > 5 and w > min_line_length and h < max_line_thickness:
                cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

        # Çizgileri orijinal görüntüden çıkar
        image_without_lines = cv2.subtract(image, mask)
        return image_without_lines

    def remove_lines_with_safe_margin(self, image, margin=15):
        """
        Çizgileri kaldırırken, sayıların olduğu alanları (çizgi çevresindeki belirli bir marjı) korur.
        :param image: Siyah-beyaz görüntü
        :param margin: Çizgi üstü ve altı korunacak piksel sayısı
        :return: Çizgileri korunmuş görüntü
        """
        # Görüntünün konturlarını bulun
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Çizgileri belirlemek için boş bir maske oluştur
        mask = np.zeros_like(image, dtype=np.uint8)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            # Çizgiyi tespit etmek için uzun ve ince konturları hedefleyin
            if aspect_ratio > 5 and h < 10:
                # Çizginin yukarı ve aşağısında belli bir marj bırak
                cv2.rectangle(mask, (x, y - margin), (x + w, y + h + margin), (255), thickness=cv2.FILLED)

        # Çizgileri orijinal görüntüden çıkar
        image_without_lines = cv2.subtract(image, mask)
        return image_without_lines
