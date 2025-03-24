import tensorflow as tf
from dataclasses import dataclass
import keras
from keras import layers, optimizers, models, ops
import os



@dataclass
class ConvParams:
    filters: int
    kernel_size: int = 3
    strides: tuple = (1, 1)
    padding: str = 'same'
    pool_size: tuple = (1, 1)

class ResBlock(layers.Layer):
    def __init__(self, convolution: layers.Conv2D | layers.SeparableConv2D, activation: str = 'relu', norm = layers.BatchNormalization(), dropout: float = 0):
        super().__init__(name = "ResBlock")
        self.convolution = convolution
        self.activation = layers.Activation(activation)
        self.norm = norm
        self.dropout = layers.Dropout(dropout)
        self.match_dim = layers.Identity()
    def build(self, input_shape):
        # Check if input and output dimensions match
        if input_shape[-1] != self.convolution.filters:
            self.match_dim = layers.Conv2D(self.convolution.filters, kernel_size=1, strides=1, padding='same')
    def call(self, x: tf.Tensor) -> tf.Tensor:
        out = self.convolution(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.match_dim(x)
        out = ops.add(x, out)
        out = self.dropout(out)
        return out

class ResNet:
    def __init__(
        self,
        in_size: tuple,
        out_shape: int,
        initial_conv: layers.Conv2D | layers.SeparableConv2D,
        initial_pool: layers.MaxPool2D,
        ResBlocks: list[ResBlock],
        fc_layers: list[layers.Dense],
        dropout: float,
        optimizer: optimizers.Optimizer,
        loss: str,
        metrics: list[str]
    ):
        self.sequential = models.Sequential()
        self.sequential.add(layers.InputLayer(shape=in_size))
        self.sequential.add(initial_conv)
        self.sequential.add(layers.BatchNormalization())
        self.sequential.add(layers.Activation('relu'))
        self.sequential.add(initial_pool)
        for block in ResBlocks:
            self.sequential.add(block)
        self.sequential.add(layers.GlobalAveragePooling2D())
        for layer in fc_layers:
            self.sequential.add(layer)
            self.sequential.add(layers.BatchNormalization())
            self.sequential.add(layers.Activation('relu'))
            self.sequential.add(layers.Dropout(dropout))
        self.sequential.add(layers.Dense(out_shape))
        
        self.sequential.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=True)
    def fit(self, *args, **kwargs):
        self.sequential.fit(*args, **kwargs)
    def save(self, path):
        self.sequential.save(path)
    def load(self, path):
        self.sequential = models.load_model(path)