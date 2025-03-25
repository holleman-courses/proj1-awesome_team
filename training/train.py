
import os
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, optimizers

from classifier import ResNet, build_res_block

import matplotlib.pyplot as plt

os.system('clear')

image_size = (176, 144)
# 112658 with separable projection
# 112402 with normal projection
# 112290 with original image size, but training is 3x slower
# 47138 with first 2 resblocks removed, 2x slower than original model

p_drop = 0.2 #prev 0.3
model = ResNet(
    in_size=(*image_size, 1),
    out_shape=1,
    initial_conv=layers.SeparableConv2D(64, 7, dilation_rate=2, padding='same', use_bias=False),
    initial_pool=layers.MaxPool2D(),
    ResBlocks=[
        #build_res_block(128, 5, dropout=p_drop),
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
        layers.Dense(64, use_bias=False),
        layers.Dense(32, use_bias=False),
        layers.Dense(16, use_bias=False),
    ],
    dropout=p_drop,
    optimizer=optimizers.Adam(learning_rate=1e-3, weight_decay=1e-5),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
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


hist = model.fit(
    save_path='model.h5',
    reset_best_acc=True, 
    x=train,
    epochs=500,
    validation_data=val,
    validation_batch_size=179
)

model_loaded = keras.models.load_model('model.h5')
model_loaded.evaluate(val)

fig, ax = plt.subplots(2)
ax[0].set_title('Loss')
ax[0].plot(hist.history['loss'], label='train')
ax[0].plot(hist.history['val_loss'], label='val')
ax[0].legend()

ax[1].set_title('Accuracy')
ax[1].plot(hist.history['accuracy'], label='train')
ax[1].plot(hist.history['val_accuracy'], label='val')
ax[1].legend()

plt.savefig('training.png')
plt.show()