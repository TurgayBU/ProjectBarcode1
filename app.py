from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from barcodes import Barcodes  # Barcodes sınıfını burada import ediyoruz

app = Flask(__name__)

# Yükleme dizini
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


# Dosya uzantısı kontrol fonksiyonu
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html', predicted_number=None, image_file=None)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Geçici hata kontrolü için dosyayı hemen kaydediyoruz
        file.save(filepath)

        try:
            barcode_processor = Barcodes(filepath)  # Barcodes sınıfını kullanıyoruz
            predicted_number = barcode_processor.process_and_predict()  # Sayı tahmini yapıyoruz

            if predicted_number == "HATA":
                return render_template('index.html', predicted_number=None, error="Hata: 6 hane bulunamadı.")

            return render_template('index.html', predicted_number=predicted_number, image_file=filename)
        except FileNotFoundError as e:
            # Model dosyası bulunamıyorsa bilgilendirici bir hata döndür
            return render_template('index.html', predicted_number=None, error="Model dosyası bulunamadı: " + str(e))
        except ValueError as e:
            # Görüntü işleme hatalarını ele alıyoruz
            return render_template('index.html', predicted_number=None, error=str(e))
    return redirect(request.url)

if __name__ == '__main__':
    app.run(port=8080, debug=True)
