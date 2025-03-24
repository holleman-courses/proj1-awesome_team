
import os
import tensorflow as tf
import keras
from keras import layers
from keras import models, optimizers

from classifier import ResBlock, ResNet

image_size = (64, 64)

model = ResNet(
    in_size=(*image_size, 1),
    out_shape=1,
    initial_conv=layers.Conv2D(64, 7, padding='same', use_bias=False),
    initial_pool=layers.MaxPool2D(),
    ResBlocks=[
        ResBlock(128, 5),
        ResBlock(64, 5),
        ResBlock(64, 5),
        ResBlock(64, 5),
        ResBlock(32, 3),
        ResBlock(32, 3),
        ResBlock(32, 3),
        ResBlock(32, 3),
        ResBlock(32, 3),
    ],
    fc_layers=[
        layers.Dense(64),
        layers.Dense(32),
        layers.Dense(16),
    ],
    dropout=0,
    optimizer=optimizers.Adam(learning_rate=1e-3, weight_decay=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

num_params = model.sequential.count_params()
print('-'*40)
print(f'Total number of parameters: {num_params}')
print('-'*40)

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



model.fit(train, epochs=30, validation_data=val)
model.save('model.h5')
