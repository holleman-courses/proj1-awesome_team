import numpy as np
import tempfile
import tensorflow as tf
from tf_keras import models, optimizers
import tf_keras as keras
import tensorflow_model_optimization as tfmot
from classifier import ResBlock, ResNet

image_size = (64, 64)

model = models.load_model('model.h5')


_, val = keras.utils.image_dataset_from_directory(
    directory = 'dataset',
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=32,
    image_size=image_size,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='both'
)

model.evaluate(val)
