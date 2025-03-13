from PIL import Image
import os
import tensorflow as tf
import keras
from keras import layers
from keras import models, optimizers

from classifier import Classifier, ConvParams

dataset = keras.utils.image_dataset_from_directory(
    directory = 'dataset/resized/dataset',
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

model = Classifier(
    in_size=(256, 256, 3),
    out_shape=1,
    conv_layers= [
        ConvParams(32),
        ConvParams(64),
        ConvParams(128),
    ],
    fc_layers=[128, 64],
    dropout=0.5,
    optimizer=optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
    
)

num_params = model.model.count_params()
print(f'Total number of parameters: {num_params}')
model.fit(dataset, epochs=15)
model.save('model.h5')
