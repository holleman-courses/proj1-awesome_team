
import os
import tensorflow as tf
import keras
from keras import layers
from keras import models, optimizers

from classifier import ResBlock, ResNet

train, val = keras.utils.image_dataset_from_directory(
    directory = 'dataset/dataset',
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=1,
    image_size=(176, 144),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='both'
)

model = ResNet(
    in_size=(176, 144),
    out_shape=1,
    initial_conv=layers.Conv2D(5, 2, padding='same', use_bias=False),
    initial_pool=layers.MaxPool2D(),
    ResBlocks=[
        ResBlock(layers.Conv2D(3, 2, padding='same', use_bias=False)),
        ResBlock(layers.Conv2D(3, 2, padding='same', use_bias=False)),
        ResBlock(layers.Conv2D(3, 2, padding='same', use_bias=False)),
        ResBlock(layers.Conv2D(3, 2, padding='same', use_bias=False)),
        ResBlock(layers.Conv2D(3, 2, padding='same', use_bias=False)),
    ],
    fc_layers=[
        layers.Dense(128),
        layers.Dense(64)
    ],
    dropout=0.3,
    optimizer=optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

num_params = model.sequential.count_params()
print(f'Total number of parameters: {num_params}')
model.fit(train, epochs=15, validation_data=val)
model.save('model.h5')
