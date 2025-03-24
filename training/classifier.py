import tensorflow as tf
from keras import layers, optimizers, models, ops



class ResBlock(layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: tuple[int] = (1,1),
        padding:str = 'same',
        activation: str = 'relu',
        dropout: int = 0
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.dropout = dropout
        
        
        self.conv1 = layers.SeparableConv2D(filters, kernel_size, strides, padding=padding, use_bias=False)
        self.conv2 = layers.SeparableConv2D(filters, kernel_size, strides, padding=padding, use_bias=False)
        self.norm1 = layers.BatchNormalization()
        self.norm2 = layers.BatchNormalization()
        self.activation1 = layers.Activation(activation)
        self.activation2 = layers.Activation(activation)
        self.dropout = layers.Dropout(dropout)
        
        self.match_dim = layers.SeparableConv2D(
            filters,
            kernel_size=1,
            strides=strides,
            padding='same',
            use_bias=False
        ) if strides != (1,1) else layers.Identity()
    
    def build(self, input_shape: tuple):
        if input_shape[-1] != self.conv1.filters:
            self.match_dim = layers.Conv2D(
                self.filters,
                kernel_size=1,
                strides=self.strides,
                padding='same',
                use_bias=False
            )
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_matched = self.match_dim(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation2(out)
        
        out = layers.Add()([out, x_matched])
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
        self.sequential.add(layers.Dense(out_shape, activation='sigmoid'))
        
        self.sequential.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=True)
    def fit(self, *args, **kwargs):
        self.sequential.fit(*args, **kwargs)
    def save(self, path):
        self.sequential.save(path)
    def load(self, path):
        self.sequential = models.load_model(path)