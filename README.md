# ProjectBarcode — Barcode Reader and Product Tracking System

A Flask application that reads 6-digit barcodes from a camera or uploaded image, uses an MNIST-based digit recognition model, and fetches product information from a database based on the scanned barcode.

## 🎯 Features

- 📷 **Live barcode reading via camera**
- 🖼️ **Barcode reading from uploaded image**
- 🔢 **MNIST-based digit recognition** (TensorFlow/Keras)
- 🗄️ **Product lookup via MySQL database**
- 👤 **Manager login**
- 📧 **Email discount notifications** (subscription system)
- 💰 **Price update and discount tracking**

## 🛠️ Technologies Used

| Layer | Technology |
|---|---|
| Web Framework | Flask |
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV |
| Database | MySQL |
| Email | smtplib (Gmail SMTP) |
| Frontend | HTML/CSS/JS (Jinja2 templates) |

## 📁 Project Structure

```
ProjectBarcode1/
├── app.py                  # Main Flask application
├── barcodes.py             # Barcode reading and digit detection
├── mnistmodel.py           # MNIST model loading and prediction
├── train_model.py          # Model training script
├── mnist_model.keras       # Trained active model
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates
│   ├── index.html
│   ├── xmanager.html
│   ├── login.html
│   └── verify.html
├── static/                 # CSS/JS files
├── uploads/                # Uploaded images (not tracked in git)
├── debug/                  # Debug images (not tracked in git)
└── archive/                # Old/experimental models (optional)
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/TurgayBU/ProjectBarcode1.git
cd ProjectBarcode1
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up the MySQL database

Create a database named `Products` and add the tables below.

> **Note:** The schemas below are representative examples. Adjust column names and types according to your actual implementation.

```sql
CREATE DATABASE Products;

USE Products;

CREATE TABLE Products (
    ProductsId VARCHAR(10) PRIMARY KEY,
    ProductName VARCHAR(255),
    Price DECIMAL(10, 2)
);

CREATE TABLE Manager (
    Id INT AUTO_INCREMENT PRIMARY KEY,
    Username VARCHAR(50),
    Password VARCHAR(50)
);

CREATE TABLE Subscribers (
    Id INT AUTO_INCREMENT PRIMARY KEY,
    Email VARCHAR(255),
    ProductId VARCHAR(10),
    VerificationCode VARCHAR(10),
    IsVerified BOOLEAN DEFAULT FALSE
);
```

### 5. Update database connection settings

In `app.py`, replace the following lines with your own MySQL credentials:

```python
connection = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="YOUR_PASSWORD",
    database="Products"
)
```

### 6. Run the application

```bash
python app.py
```

Open in your browser: `http://localhost:8080`

## 🎮 Usage

### User (Customer)

1. On the home page, either **upload a photo** or **open the camera**
2. The barcode is read, product information and price are displayed
3. You can subscribe via email for **discount notifications**

### Manager (Admin)

1. Log in at `/login`
2. Update product prices
3. When a price drops, subscribed users are automatically notified by email

## 🔬 Model Training

If you want to retrain the model:

```bash
python train_model.py
```

This script:

- Downloads the MNIST dataset
- Trains a CNN model (BatchNormalization + Dropout)
- Saves the best model as `mnist_model.keras`

## ⚙️ Configuration

| Setting | File | Variable |
|---|---|---|
| Flask port | `app.py` | `app.run(port=8080)` |
| Database credentials | `app.py` | `connection = mysql.connector.connect(...)` |
| Email sender | `app.py` | `sender_email`, `sender_password` |
| Model file path | `mnistmodel.py` | `model_path='mnist_model.keras'` |

> ⚠️ **Security Note:** Sensitive information such as email passwords and database passwords should be moved to environment variables (`.env` file).

## 📌 Notes

- The `debug/` and `uploads/` folders are created automatically at runtime
- Model prediction works on 28x28 grayscale images
- The barcode must be 6 digits long (otherwise "ERROR" is returned)

## 👤 Developer

**TurgayBU** — [GitHub](https://github.com/TurgayBU)

## 📄 License

This project currently has no license specified. You can add an MIT license if you wish.