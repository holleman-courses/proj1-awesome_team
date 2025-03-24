import numpy as np
import tempfile
import tensorflow as tf
from tf_keras import models, optimizers, layers
import tf_keras as keras
import tensorflow_model_optimization as tfmot
from classifier import ResNet

image_size = (64, 64)



model = models.load_model('model.h5')
print('Begin Quantization')
model = tfmot.quantization.keras.quantize_model(model)
print("End Quantization")
model.save('model_qat.h5')
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])


train, val = keras.utils.image_dataset_from_directory(
    directory = 'dataset',
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=16,
    image_size=image_size,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='both'
)
model.eval(val)
