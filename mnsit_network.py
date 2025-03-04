import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def load_and_prepare_data():
    """
    MNIST veri kümesini yükler ve işleme sokar.
    """
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    return x_train, y_train, x_test, y_test


def build_model():
    """
    Gelişmiş bir CNN modeli oluşturur.
    """
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_training_metrics(history):
    """
    Modelin eğitim sonuçlarını grafiğe döker.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.title('Doğruluk')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Kayıp')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def train_and_evaluate_model():
    """
    Modeli eğitir ve değerlendirir.
    """
    x_train, y_train, x_test, y_test = load_and_prepare_data()
    model = build_model()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("mnist_model.keras", save_best_only=True)
    ]

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=15, batch_size=64, callbacks=callbacks)

    model.save("mnist_model.keras")
    print("Model başarıyla kaydedildi!")

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Doğruluğu: %{test_accuracy * 100:.2f}")

    plot_training_metrics(history)
    return model, x_test, y_test


def predict_sample(model, x_test, y_test):
    """
    Rastgele bir örnekle tahmin yapar.
    """
    sample_idx = np.random.randint(0, len(x_test))
    sample_image = x_test[sample_idx]
    sample_label = y_test[sample_idx]

    plt.imshow(sample_image.squeeze(), cmap="Greys")
    plt.title(f"Gerçek: {sample_label}")
    plt.show()

    prediction = np.argmax(model.predict(sample_image.reshape(1, 28, 28, 1)))
    print(f"Model Tahmini: {prediction}")


if __name__ == "__main__":
    trained_model, x_test, y_test = train_and_evaluate_model()
    predict_sample(trained_model, x_test, y_test)
