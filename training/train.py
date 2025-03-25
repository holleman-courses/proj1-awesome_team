
import os
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, optimizers

from classifier import ResNet, build_res_block

import matplotlib.pyplot as plt

image_size = (64, 64)

p_drop = 0.4 #prev 0.3
model = ResNet(
    in_size=(*image_size, 1),
    out_shape=1,
    initial_conv=layers.Conv2D(64, 7, padding='same', use_bias=False),
    initial_pool=layers.MaxPool2D(),
    ResBlocks=[
        build_res_block(128, 5, dropout=p_drop),
        build_res_block(64, 5, dropout=p_drop),
        build_res_block(64, 5, dropout=p_drop),
        build_res_block(64, 5, dropout=p_drop),
        build_res_block(32, 3, dropout=p_drop),
        build_res_block(32, 3, dropout=p_drop),
        build_res_block(32, 3, dropout=p_drop),
        build_res_block(32, 3, dropout=p_drop),
        build_res_block(32, 3, dropout=p_drop),
    ],
    fc_layers=[
        layers.Dense(64),
        layers.Dense(32),
        layers.Dense(16),
    ],
    dropout=p_drop,
    optimizer=optimizers.Adam(learning_rate=1e-3, weight_decay=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

train, val = keras.utils.image_dataset_from_directory(
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


hist = model.fit('model.h5', x=train, epochs=1, validation_data=val, validation_batch_size=179)

model_loaded = keras.models.load_model('model.h5')
model_loaded.evaluate(val)

plt.title('Loss')
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='val')
plt.legend()
plt.savefig('loss.png')

plt.title('Accuracy')
plt.plot(hist.history['accuracy'], label='train')
plt.plot(hist.history['val_accuracy'], label='val')
plt.legend()
plt.savefig('accuracy.png')

