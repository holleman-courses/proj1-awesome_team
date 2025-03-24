import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from person import mobilenet_v1  # Import MobileNetV1 model

# Define parameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.00001

def load_images_from_directory(directory, label):
    images, labels = [], []
    for img_file in os.listdir(directory):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(directory, img_file)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(label)
    return images, labels

def load_data(data_dir):
    pos_images, pos_labels = load_images_from_directory(os.path.join(data_dir, 'positive'), 1)
    neg_images, neg_labels = load_images_from_directory(os.path.join(data_dir, 'negative'), 0)

    images = np.array(pos_images + neg_images) / 255.0  # Normalize
    labels = np.array(pos_labels + neg_labels)
    return images, labels

def plot_training_history(history, output_dir):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))

def convert_to_tflite(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

def save_model_and_metrics(model, history, x_train, y_train, x_val, y_val, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, 'model.h5'))
    convert_to_tflite(model, os.path.join(output_dir, 'model.tflite'))

    # Evaluate model
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)

    # Save metrics
    with open(os.path.join(output_dir, 'model_metrics.txt'), 'w') as f:
        f.write(f"Training Accuracy: {train_acc:.4f}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Positive Samples: {np.sum(y_train) + np.sum(y_val)}\n")
        f.write(f"Negative Samples: {len(y_train) + len(y_val) - np.sum(y_train) - np.sum(y_val)}\n")

    plot_training_history(history, output_dir)

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    output_dir = os.path.join(os.path.dirname(__file__), 'Output')

    # Load and split data
    print("Loading data...")
    images, labels = load_data(data_dir)
    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")

    # Load MobileNetV1 model
    print("Building MobileNetV1 model...")
    model = mobilenet_v1(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=1, num_filters=8)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    print("Training model...")
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val), verbose=1)

    # Save results
    save_model_and_metrics(model, history, x_train, y_train, x_val, y_val, output_dir)
    print(f"Model and metrics saved in {output_dir}")

if __name__ == "__main__":
    main()
