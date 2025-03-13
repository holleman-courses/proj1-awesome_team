import tensorflow as tf
from dataclasses import dataclass
import keras
from keras import layers, optimizers, models
layers.Conv2D
import os

@dataclass
class ConvParams:
    filters: int
    kernel_size: int = 3
    strides: tuple = (1, 1)
    padding: str = 'same'
    pool_size: tuple = (1, 1)

class Classifier:
    def __init__(self, in_size, out_shape, conv_layers: list[ConvParams], fc_layers: list[int], dropout, optimizer, loss, metrics):
        self.sequential = models.Sequential()
        self.sequential.add(layers.InputLayer(shape=in_size))
        for layer in conv_layers:
            self.sequential.add(layers.Conv2D(layer.filters, layer.kernel_size, layer.strides, layer.padding))
            self.sequential.add(layers.BatchNormalization())
            self.sequential.add(layers.Activation('relu'))
            self.sequential.add(layers.MaxPooling2D(pool_size=layer.pool_size))
            self.sequential.add(layers.Dropout(dropout))
        self.sequential.add(layers.Flatten())
        for layer in fc_layers:
            self.sequential.add(layers.Dense(layer))
            self.sequential.add(layers.BatchNormalization())
            self.sequential.add(layers.Activation('relu'))
            self.sequential.add(layers.Dropout(dropout))
        self.sequential.add(layers.Dense(out_shape))
        
        self.sequential.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    def fit(self, *args, **kwargs):
        self.sequential.fit(*args, **kwargs)
    def save(self, path):
        self.sequential.save(path)
    def load(self, path):
        self.sequential = models.load_model(path)