
import os
import tensorflow as tf
import keras
from keras import layers
from keras import models, optimizers

from classifier import Classifier, ConvParams

train, val = keras.utils.image_dataset_from_directory(
    directory = 'dataset/resized/dataset',
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    batch_size=1,
    image_size=(64, 64),
    shuffle=True,
    seed=42,
    validation_split=0.4,
    subset='both'
)

model = Classifier(
    in_size=(64, 64, 3),
    out_shape=1,
    conv_layers= [
        ConvParams(8),
        #ConvParams(16),
        #ConvParams(32),
        #ConvParams(64),
        #ConvParams(128),
    ],
    fc_layers=[128],
    dropout=0.3,
    optimizer=optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
    
)

num_params = model.sequential.count_params()
print(f'Total number of parameters: {num_params}')
model.fit(train, epochs=15, validation_data=val)
model.save('model.h5')
