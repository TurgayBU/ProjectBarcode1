import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response, flash
import os
from werkzeug.utils import secure_filename
from barcodes import Barcodes  # Barcodes sınıfını kullanmak için
import mysql.connector
from flask import session
import base64
from io import BytesIO
import threading
from PIL import Image
stop_camera = False


app = Flask(__name__)
app.secret_key = 'çok-gizli-bir-anahtar'
connection = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="Yeni0000",
    database="Products"
)
def generate_camera_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame= cv2.flip(frame,1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Yükleme ve statik debug dizinleri
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DEBUG_FOLDER'] = 'debug'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


# Dosya uzantısı kontrol fonksiyonu
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def prediction_with_camera():
    global stop_camera
    stop_camera = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return None

    predicted_number = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        try:
            predictor = Barcodes(frame)
            predicted_number = predictor.process_and_predict()
            print("Tahmin:", predicted_number)
        except Exception as e:
            print("İşlenemedi:", e)

        if stop_camera:
            break

    cap.release()
    cv2.destroyAllWindows()
    return predicted_number

@app.route('/')
def index():
    return render_template('index.html', predicted_number=None, image_file=None, debug_image_file=None,is_manager=session.get('manager_logged_in', False),product=None)
@app.route('/cameraman')
def cameraman():
    predicted_number = prediction_with_camera()

    if predicted_number is None:
        return render_template('index.html', error="Could not taken guess.")

    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Products WHERE ProductsId=%s", (predicted_number,))
    product = cursor.fetchone()
    cursor.close()

    return render_template(
        'index.html',
        predicted_number=predicted_number,
        product=product,
        debug_image_file='contours_debug.jpg' if os.path.exists(os.path.join(app.config['DEBUG_FOLDER'], 'contours_debug.jpg')) else None,
        is_manager=session.get('manager_logged_in', False)
    )

@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' in request.files:
        # Normal dosya yükleme
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # Yüklenen dosyayı kaydetme
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Barcode tahmin işlemi
            try:
                barcode_processor = Barcodes(filepath)
                predicted_number = barcode_processor.process_and_predict()

                debug_image_path = os.path.join(app.config['DEBUG_FOLDER'], 'contours_debug.jpg')
                if not os.path.exists(debug_image_path):
                    debug_image_path = None

                if predicted_number == "ERROR":
                    return render_template(
                        'index.html',
                        predicted_number=None,
                        error="Error: 6 digit contour not found. Please try again.",
                        debug_image_file='contours_debug.jpg' if debug_image_path else None
                    )

                cursor = connection.cursor(dictionary=True)
                cursor.execute("SELECT * FROM Products WHERE ProductsId=%s", (predicted_number,))
                product = cursor.fetchone()
                cursor.close()

                return render_template(
                    'index.html',
                    predicted_number=predicted_number,
                    image_file=filename,
                    debug_image_file='contours_debug.jpg' if debug_image_path else None,
                    product=product,
                    is_manager=session.get('manager_logged_in', False)
                )

            except Exception as e:
                return render_template('index.html', predicted_number=None, error=str(e))

    elif 'image_data' in request.form:
        # Kamera fotoğrafı
        image_data = request.form['image_data']

        if not image_data:
            return redirect(request.url)

        # 'data:image/jpeg;base64,' kısmını çıkar
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        # Resmi PIL ile açma
        image = Image.open(BytesIO(image_bytes))

        # Geçici olarak kaydetme
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.jpg')
        image.save(filepath)

        # Barcode tahmin işlemi
        try:
            barcode_processor = Barcodes(filepath)
            predicted_number = barcode_processor.process_and_predict()

            debug_image_path = os.path.join(app.config['DEBUG_FOLDER'], 'contours_debug.jpg')
            if not os.path.exists(debug_image_path):
                debug_image_path = None

            if predicted_number == "ERROR":
                return render_template(
                    'index.html',
                    predicted_number=None,
                    error="Error: 6 digit contour not found. Please try again.",
                    debug_image_file='contours_debug.jpg' if debug_image_path else None
                )

            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM Products WHERE ProductsId=%s", (predicted_number,))
            product = cursor.fetchone()
            cursor.close()

            return render_template(
                'index.html',
                predicted_number=predicted_number,
                image_file='captured_image.jpg',
                debug_image_file='contours_debug.jpg' if debug_image_path else None,
                product=product,
                is_manager=session.get('manager_logged_in', False)
            )

        except Exception as e:
            return render_template('index.html', predicted_number=None, error=str(e))

    return redirect(request.url)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM Manager WHERE Username=%s AND Password=%s", (username, password))
        manager = cursor.fetchone()
        cursor.close()

        if manager:
            session['manager_logged_in'] = True
            session['manager_username'] = manager['Username']  # İstersen kullanırsın
            return redirect(url_for('index'))
        else:
            flash("Login failed. Please try again.", "danger")
            return render_template('login.html')
    return render_template('login.html')

@app.route('/update_single_price', methods=['POST'])
def update_single_price():
    if not session.get('manager_logged_in'):
        return redirect(url_for('login'))

    product_id = request.form['product_id']
    new_price = request.form['new_price']

    cursor = connection.cursor()
    cursor.execute("UPDATE Products SET Price=%s WHERE ProductsId=%s", (new_price, product_id))
    connection.commit()
    cursor.close()

    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('manager_logged_in', None)
    return redirect(url_for('index'))
# Statik debug dosyasını sunmak için route
@app.route('/debug/<path:filename>')
def serve_debug_file(filename):
    return send_from_directory(app.config['DEBUG_FOLDER'], filename)
@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/stop_camera', endpoint='stop_camera_route')
def stop_camera_route():
    global stop_camera
    stop_camera = True
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(port=8080, debug=True)
