from PIL import Image
import os
import tensorflow as tf
import keras
from keras import layers
from keras import models, optimizers

dataset = keras.utils.image_dataset_from_directory(
    './dataset/resized',
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    batch_size=None,
    image_size=(256, 256),
    shuffle=True,
    seed=42,
    validation_split=0.4,
    subset='training'
)

model = models.Sequential()