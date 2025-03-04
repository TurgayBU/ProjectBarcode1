import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

class MnistModel:
    def __init__(self,model_path='mnist_model.keras'):
        import os
        os.makedirs("debug", exist_ok=True)
        self.model = self.load_model(model_path)
        self.numbers = []

    @staticmethod
    def load_model(model_path):
        try:
            print(f"Model yükleniyor: {model_path}")
            model = load_model(model_path)
            print("Model başarıyla yüklendi.")
            return model
        except Exception as e:
            print(f"Model yükleme hatası: {e}")  # Hata durumunu loglayın
            raise

    def predict_number(self, image):
        input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, axis=0)  # Model için batch boyutu ekle
        prediction = self.model.predict(input_tensor)
        print(f"Model çıktısı (olasılıklar): {prediction}")
        predicted_number = np.argmax(prediction)
        return predicted_number

#   def authenticate_number(self, resized_image):
#        normalized_image = np.expand_dims(resized_image, axis=0)
#        predictions = self.model.predict(normalized_image)
#        predicted_number = np.argmax(predictions)
#        confidence = np.max(predictions)

        # Tahmini yazdır (isteğe bağlı)
#        print(f"Tahmin: {predicted_number}, Güven: {confidence}")

        # Düşük güvenli tahminleri sadece bildir
#        if confidence < self.confidence_threshold:
#            print(f"Düşük güvenli tahmin tespit edildi! Tahmin: {predicted_number}, Güven: {confidence}")
    def set_vector(self, numbers):
        self.numbers = numbers
