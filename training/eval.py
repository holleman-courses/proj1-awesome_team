
import os
import tensorflow as tf
import keras
from keras import layers
from keras import models, optimizers
from classifier import ResBlock, ResNet



image_size = (64, 64)

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

model = keras.models.load_model('model.h5')
model.evaluate(val)
