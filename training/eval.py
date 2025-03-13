
import os
import tensorflow as tf
import keras
from keras import layers
from keras import models, optimizers

from classifier import Classifier, ConvParams

data = keras.utils.image_dataset_from_directory(
    directory = 'dataset/resized/dataset',
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    batch_size=1,
    image_size=(64, 64),
    shuffle=True,
    seed=42,
)

model = keras.models.load_model('model.h5')
model.evaluate(data)
