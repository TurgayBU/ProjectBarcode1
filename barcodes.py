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
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image format not suitable for MNIST. Try again with a different image format.")
        print(f"Image '{image_path}' successfully loaded..")
        inverted_image = cv2.bitwise_not(image)
        self.image = self.make_black_white(inverted_image)

    def make_black_white(self, image):
        print("Image converted to black and white..")
        try:
            # Global Threshold yöntemi ile daha net beyaz/siyah dönüşümü sağlanır
            _, bw_image = cv2.threshold(
                image,
                127,  # Piksel değerlerinin eşik noktası
                255,  # Eşik üstündeki piksel beyaz yapma
                cv2.THRESH_BINARY_INV  # Siyah/Beyaz ters çevrilir (iç beyaz olması için)
            )
        except Exception as e:
            print(f"An error occurred during thresholding operation: {e}")
            raise ValueError("Error occurred during black and white rendering.")

        # Görüntüyü temizleme (gereksiz gürültüleri kaldırmak için)
        return self.clean_image(bw_image)

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
            raise ValueError("Not found any contour in the image. Please check the image and try again.")

        print(f"Total contour number: {len(contours)}")

        # Konturları alanlarına (boyutlarına) göre azalan sıralama
        sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

        # En büyük 6 konturu alıyok
        top_contours = sorted_contours[:6]

        # Konturları soldan sağa sıralıyok (gerekiyorsa)
        top_contours = sorted(top_contours, key=lambda c: cv2.boundingRect(c)[0])

        # Konturları işleme
        self.numbers = []
        image_height, image_width = image.shape[:2]
        for contour in top_contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Kontura çerçeve çiz ve kaydet (debug için)
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w += padding * 4
            h += padding * 4
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Mavi çerçeve
            cv2.imwrite("debug/contours_debug.jpg", image)

            # Konturu işleme alma
            self.make_bound_rect(contour)

        if len(self.numbers) != 6:
            print(f"Error: 6 contours not founded. Founded contours: {len(self.numbers)}.")

    def make_bound_rect(self, contour):
        # Konturdan bounding box alma
        x, y, w, h = cv2.boundingRect(contour)

        # Görüntüyü belirlenen çerçeveden kesme (boşluk bırakıyoruz ki model tanısın)
        cropped_image = self.image[y:y + h, x:x + w]

        # Siyah alanları temizle (aktif kontur alanını belirle ve kes)
        bounding_box = cv2.boundingRect(cv2.findNonZero(cropped_image))  # Siyah alanı kaldır
        bx, by, bw, bh = bounding_box
        cropped_image = cropped_image[by:by + bh, bx:bx + bw]

        # Görüntünün kenarına siyah boşluk eklemek için padding ekleme
        padding = 20  # Her kenardan 20 piksel ekleyerek boşluk bırakıyorum ki model adam akıllı tanısın

        # Siyah boşluk ekleme
        padded_image = cv2.copyMakeBorder(
            cropped_image,  # Kesilen görüntü
            top=padding,  # Üst tarafa boşluk
            bottom=padding,  # Alt tarafa boşluk
            left=padding,  # Sol tarafa boşluk
            right=padding,  # Sağ tarafa boşluk hep boşluk hep boşluk
            borderType=cv2.BORDER_CONSTANT,  # Sabit renkli çerçeve türü
            value=[0, 0, 0]  # Siyah (RGB siyah renk: [0,0,0])
        )

        # Siyah boşluk eklenmiş işlenmiş görüntünün debug için kaydedilmesi
        debug_path = f"debug/padded_{x}_{y}.jpg"
        os.makedirs("debug", exist_ok=True)
        cv2.imwrite(debug_path, padded_image)
        print(f"I added the debug image on: {debug_path}")

        # Görseli yeniden boyutlandırma ve modelle tahmin ettirme
        resized_image = self.digit_resize(padded_image)
        self.numbers.append(self.predict_number(resized_image))

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
