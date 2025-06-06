import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response, flash
import os
import random
import smtplib
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
@app.route('/xmanager')
def xmanager():
    return render_template('xmanager.html', predicted_number=None, image_file=None, debug_image_file=None,is_manager=session.get('manager_logged_in', True),product=None)

@app.route('/')
def index():
    return render_template('index.html', predicted_number=None, image_file=None, debug_image_file=None,is_manager=session.get('manager_logged_in', False),product=None)
@app.route('/cameraman')
@app.route('/cameraman')
def cameraman():
    predicted_number = prediction_with_camera()
    is_manager = session.get('manager_logged_in', False)
    render_target = 'xmanager.html' if is_manager else 'index.html'

    if predicted_number is None:
        return render_template(render_target, error="Could not taken guess.")

    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Products WHERE ProductsId=%s", (predicted_number,))
    product = cursor.fetchone()
    cursor.close()

    return render_template(
        render_target,
        predicted_number=predicted_number,
        product=product,
        debug_image_file='contours_debug.jpg' if os.path.exists(os.path.join(app.config['DEBUG_FOLDER'], 'contours_debug.jpg')) else None,
        is_manager=is_manager
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

                is_manager = session.get('manager_logged_in', False)
                render_target = 'xmanager.html' if is_manager else 'index.html'

                return render_template(
                    render_target,
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

            is_manager = session.get('manager_logged_in', False)
            render_target = 'xmanager.html' if is_manager else 'index.html'

            return render_template(
                render_target,
                predicted_number=predicted_number,
                image_file='captured_image.jpg',
                debug_image_file='contours_debug.jpg' if debug_image_path else None,
                product=product,
                is_manager=is_manager
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
            return redirect(url_for('xmanager'))
        else:
            flash("Login failed. Please try again.", "danger")
            return render_template('login.html')
    return render_template('login.html')
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
@app.route('/stop_camera')
def stop_camera_route():
    global stop_camera
    stop_camera = True
    is_manager = session.get('manager_logged_in', False)
    if is_manager:
        return redirect(url_for('xmanager'))  # xmanager adındaki route'a yönlendir
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(port=8080, debug=True)

def send_verification_email(email, code):
    sender_email = "mongrosssupermarket@gmail.com"
    sender_password = "wmltunbgkovbxucs"  # App password (boşluksuz olmalı)

    subject = "Product Track Code"
    body = f"Authentication Code: {code}"
    message = f"Subject: {subject}\n\n{body}"

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(sender_email, sender_password)
            smtp.sendmail(sender_email, email, message)
            print("Verification email sent successfully.")
    except Exception as e:
        print("Failed to send email:", e)
@app.route('/subscribe_discount', methods=['POST'])
def subscribe_discount():
    email = request.form['email']
    product_id = request.form['product_id']
    code = str(random.randint(100000, 999999))

    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Subscribers WHERE Email=%s AND ProductId=%s", (email, product_id))
    existing = cursor.fetchone()

    if existing:
        # Zaten varsa, kodu güncelle
        cursor.execute("UPDATE Subscribers SET VerificationCode=%s WHERE Id=%s", (code, existing['Id']))
    else:
        cursor.execute("INSERT INTO Subscribers (Email, ProductId, VerificationCode) VALUES (%s, %s, %s)", (email, product_id, code))
    connection.commit()
    cursor.close()

    send_verification_email(email, code)
    flash("Authentication code has been sended to your e-email!", "info")
    return redirect(url_for('verify_email', email=email, product_id=product_id))
@app.route('/verify_email', methods=['GET', 'POST'])
def verify_email():
    if request.method == 'GET':
        email = request.args.get('email')
        product_id = request.args.get('product_id')
        return render_template('verify.html', email=email, product_id=product_id)

    # POST işlemi
    email = request.form['email']
    product_id = request.form['product_id']
    entered_code = str(request.form['code'])  # Güvenli şekilde string yap

    cursor = connection.cursor()
    cursor.execute("SELECT VerificationCode FROM Subscribers WHERE Email=%s AND ProductId=%s", (email, product_id))
    result = cursor.fetchone()

    if result and result[0] == entered_code:
        cursor.execute("UPDATE Subscribers SET IsVerified=TRUE WHERE Email=%s AND ProductId=%s", (email, product_id))
        connection.commit()
        flash("E-mail has been authenticated successfully and added to the discount system.", "success")
        cursor.close()
        return redirect(url_for('index'))
    else:
        cursor.close()
        flash("Code is wrong. Please try again.", "danger")
        return render_template('verify.html', email=email, product_id=product_id)
@app.route('/update_single_price', methods=['POST'])
def update_single_price():
    product_id = request.form['product_id']
    new_price = float(request.form['new_price'])

    cursor = connection.cursor()

    # Önce eski fiyatı çek
    cursor.execute("SELECT Price FROM Products WHERE ProductsId = %s", (product_id,))
    old_price_result = cursor.fetchone()  # İlk sorgunun sonucunu çekiyoruz!

    # Sonra ürün adını çek
    cursor.execute("SELECT ProductName FROM Products WHERE ProductsId = %s", (product_id,))
    product_name_row = cursor.fetchone()
    product_name = product_name_row[0] if product_name_row else None

    if old_price_result:
        old_price = old_price_result[0]

        # Eğer fiyat düştüyse
        if new_price < old_price:
            # Ürünün fiyatını güncelle
            cursor.execute("UPDATE Products SET Price = %s WHERE ProductsId = %s", (new_price, product_id))
            connection.commit()

            # İndirimi takip edenleri çek
            cursor.execute("SELECT Email FROM Subscribers WHERE ProductId = %s AND IsVerified = TRUE", (product_id,))
            subscribers = cursor.fetchall()

            # Her takipçiye mesaj gönder
            for (email,) in subscribers:
                send_discount_notification(email, product_name, old_price, new_price)

            flash("Price updated and notifications sent to subscribers.", "success")

        else:
            # Fiyat düşmediyse sadece güncelle
            cursor.execute("UPDATE Products SET Price = %s WHERE ProductsId = %s", (new_price, product_id))
            connection.commit()
            flash("Price updated.", "success")
    else:
        flash("Product not found.", "danger")

    cursor.close()
    return redirect(url_for('index'))  # Güncelleme sonrası yönlendirilecek sayfa
def send_discount_notification(email, product_id, old_price, new_price):
    subject = "Price Drop Alert!"
    body = f"The price of {product_id} has dropped from {old_price} to {new_price}. Check it out now!"
    msg = f"Subject: {subject}\n\n{body}"

    sender_email = "mongrosssupermarket@gmail.com"
    sender_password = "wmltunbgkovbxucs"
    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(sender_email, sender_password)
        smtp.sendmail(sender_email, email, msg)


