from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from barcodes import Barcodes  # Barcodes sınıfını kullanmak için

app = Flask(__name__)

# Yükleme ve statik debug dizinleri
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DEBUG_FOLDER'] = 'debug'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


# Dosya uzantısı kontrol fonksiyonu
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html', predicted_number=None, image_file=None, debug_image_file=None)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Kaydedilen dosyanın yolu
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Dosyayı yükleme dizinine kaydetme
        file.save(filepath)

        try:
            # Tahmin işlemini gerçekleştirme
            barcode_processor = Barcodes(filepath)  # Barcodes sınıfını kullanıyoruz
            predicted_number = barcode_processor.process_and_predict()  # Sayı tahmini yapıyoruz

            # Debug görüntü dosyasının yolu
            debug_image_path = os.path.join(app.config['DEBUG_FOLDER'], 'contours_debug.jpg')

            if not os.path.exists(debug_image_path):
                debug_image_path = None  # Eğer contours_debug.jpg yoksa, None döndür

            if predicted_number == "ERROR":
                return render_template(
                    'index.html',
                    predicted_number=None,
                    error="Error: 6 digit contour not found. Please try again.",
                    debug_image_file='contours_debug.jpg' if debug_image_path else None
                )

            return render_template(
                'index.html',
                predicted_number=predicted_number,
                image_file=filename,
                debug_image_file='contours_debug.jpg' if debug_image_path else None
            )
        except FileNotFoundError as e:
            # Model dosyası bulunamadığı durumda hata gösterme
            return render_template('index.html', predicted_number=None, error="Model file not founded: " + str(e))
        except ValueError as e:
            # Görüntü işleme hatalarını ele alıyoz
            return render_template('index.html', predicted_number=None, error=str(e))
    return redirect(request.url)


# Statik debug dosyasını sunmak için route
@app.route('/debug/<path:filename>')
def serve_debug_file(filename):
    return send_from_directory(app.config['DEBUG_FOLDER'], filename)


if __name__ == '__main__':
    app.run(port=8080, debug=True)
