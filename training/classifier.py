import tensorflow as tf
from dataclasses import dataclass
import keras
from keras import layers, optimizers, models
layers.Conv2D
from PIL import Image
import os

@dataclass
class ConvParams:
    filters: int
    kernel_size: int = 3
    strides: tuple = (1, 1)
    padding: str = 'same'
    pool_size: tuple = (1, 1)

class Classifier:
    def __init__(self, in_size, out_shape, conv_layers: list[ConvParams], fc_layers, dropout, optimizer, loss, metrics):
        self.model = models.Sequential()
        self.model.add(layers.InputLayer(input_shape=in_size))
        for layer in conv_layers:
            self.model.add(layers.Conv2D(layer.filters, layer.kernel_size, layer.strides, layer.padding))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.MaxPooling2D(pool_size=layer.pool_size))
            self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Flatten())
        # dummy_in = keras.KerasTensor(shape=(None, in_size), dtype=tf.float32)
        # dummy_out = self.model(dummy_in)
        # fc_in = dummy_out.shape[1]
        for layer in fc_layers:
            self.model.add(layers.Dense(layer))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Dropout(dropout))
        self.model.add(layers.Dense(out_shape))
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)