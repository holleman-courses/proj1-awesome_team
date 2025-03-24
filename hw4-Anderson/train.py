import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Activation, Flatten, BatchNormalization,
                                     Conv2D, MaxPooling2D)
from tensorflow.keras.regularizers import l2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from person import mobilenet_v1

# Define parameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 2  # Binary classification
LEARNING_RATE = 0.0008
L2_REG = 1e-4

# 0.00001

def loadImagesFromDirectory(directory, label):
    images, labels = [], []
    for imgFile in os.listdir(directory):
        if imgFile.endswith(('.jpg', '.jpeg', '.png')):
            imgPath = os.path.join(directory, imgFile)
            img = tf.keras.preprocessing.image.load_img(imgPath, target_size=(IMG_SIZE, IMG_SIZE))
            imgArray = tf.keras.preprocessing.image.img_to_array(img)
            images.append(imgArray)
            labels.append(label)
    return images, labels

def loadData(dataDir):
    posImages, posLabels = loadImagesFromDirectory(os.path.join(dataDir, 'positive'), 1)
    negImages, negLabels = loadImagesFromDirectory(os.path.join(dataDir, 'negative'), 0)
    
    images = np.array(posImages + negImages) / 255.0  # Normalize
    labels = np.array(posLabels + negLabels)
    return images, labels

def buildCnnModel():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    
    # Convolutional Blocks
    for filters in [32, 64, 128]:
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(L2_REG))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Fully Connected Layers
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(L2_REG))(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def plotTrainingHistory(history, outputDir):
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
    plt.savefig(os.path.join(outputDir, 'trainingHistory.png'))

def convertToTflite(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tfliteModel = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tfliteModel)

def saveModelAndMetrics(model, history, x_train, y_train, x_val, y_val, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, 'model.h5'))
    convertToTflite(model, os.path.join(output_dir, 'model.tflite'))

    # Evaluate model
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)

    # Save metrics
    with open(os.path.join(output_dir, 'model_metrics.txt'), 'w') as f:
        f.write(f"Training Accuracy: {train_acc:.4f}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Positive Samples: {np.sum(y_train) + np.sum(y_val)}\n")
        f.write(f"Negative Samples: {len(y_train) + len(y_val) - np.sum(y_train) - np.sum(y_val)}\n")

    plotTrainingHistory(history, output_dir)

def main():
    dataDir = os.path.join(os.path.dirname(__file__), 'dataset')
    outputDir = os.path.join(os.path.dirname(__file__), 'Output')
    
    # Load and split data
    print("Loading data...")
    images, labels = loadData(dataDir)
    xTrain, xVal, yTrain, yVal = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
    
    print(f"Training samples: {len(xTrain)}, Validation samples: {len(xVal)}")
    
    # Build and train model
    print("Building model...")
    model = buildCnnModel()
    model.summary()
    
    print("Training model...")
    history = model.fit(xTrain, yTrain, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(xVal, yVal), verbose=1)
    
    # Save results
    saveModelAndMetrics(model, history, xTrain, yTrain, xVal, yVal, outputDir)
    print(f"Model and metrics saved in {outputDir}")

if __name__ == "__main__":
    main()
