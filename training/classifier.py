import tensorflow as tf
from tf_keras import layers, optimizers, models
import tf_keras as keras


def build_res_block(filters, kernel_size=3, strides=(1,1), padding='same', activation='relu', dropout=0):
    def res_block(x):
        shortcut = x
        
        out = layers.SeparableConv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        out = layers.BatchNormalization()(out)
        out = layers.Activation(activation)(out)
        
        out = layers.SeparableConv2D(filters, kernel_size, strides=(1,1), padding=padding, use_bias=False)(out)
        out = layers.BatchNormalization()(out)
        out = layers.Activation(activation)(out)
        
        if strides != (1,1) or x.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, kernel_size=1, strides=strides, padding='same', use_bias=False)(x)
            shortcut = layers.BatchNormalization()(shortcut)
        
        out = layers.Add()([out, shortcut])
        out = layers.Dropout(dropout)(out)
        return out
    return res_block



# @keras.saving.register_keras_serializable(package="ResNet")

# class ResBlock(layers.Layer):    
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
#     def __init__(
#         self,
#         filters: int, 
#         kernel_size: int = 3,
#         strides: tuple[int] = (1,1),
#         padding:str = 'same',
#         activation: str = 'relu',
#         dropout: int = 0,
#         use_projection: bool = None,
#         base_args = [],
#         base_kwargs = {},
#     ):
#         super().__init__(*base_args, **base_kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.padding = padding
#         self.activation = activation
#         self.dropout_p = dropout
        
        
#         self.conv1 = layers.SeparableConv2D(filters, kernel_size, strides, padding=padding, use_bias=False)
#         self.conv2 = layers.SeparableConv2D(filters, kernel_size, strides, padding=padding, use_bias=False)
#         self.norm1 = layers.BatchNormalization()
#         self.norm2 = layers.BatchNormalization()
#         self.activation1 = layers.Activation(activation)
#         self.activation2 = layers.Activation(activation)
#         self.dropout = layers.Dropout(dropout)
        
#         if use_projection is None:
#             use_projection = strides != (1,1)
#         self.use_projection = use_projection
        
#         self.match_dim = layers.SeparableConv2D(
#             filters,
#             kernel_size=1,
#             strides=strides,
#             padding='same',
#             use_bias=False
#         ) if self.use_projection else layers.Identity()
    
#     # def build(self, input_shape: tuple):
#     #     if self.use_projection:
#     #         return
#     #     if input_shape[-1] != self.filters:
#     #         self.match_dim = layers.Conv2D(
#     #             self.filters,
#     #             kernel_size=1,
#     #             strides=self.strides,
#     #             padding='same',
#     #             use_bias=False
#     #         )
#     #         self.use_projection = True
            
        
#     def call(self, x: tf.Tensor) -> tf.Tensor:
#         x_matched = self.match_dim(x)
        
#         out = self.conv1(x)
#         out = self.norm1(out)
#         out = self.activation1(out)
        
#         out = self.conv2(out)
#         out = self.norm2(out)
#         out = self.activation2(out)
        
#         out = layers.Add()([out, x_matched])
#         out = self.dropout(out)
        
#         return out

#     def get_config(self):
#         base_config = super().get_config()
#         return {
#             'base_kwargs': base_config,
#             'filters': self.filters,
#             'kernel_size': self.kernel_size,
#             'strides': self.strides,
#             'padding': self.padding,
#             'activation': self.activation,
#             'dropout': self.dropout_p,
#             'use_projection': self.use_projection
#         }

class ResNet:
    def __init__(
        self,
        in_size: tuple,
        out_shape: int,
        initial_conv: layers.Conv2D | layers.SeparableConv2D,
        initial_pool: layers.MaxPool2D,
        ResBlocks: list,
        fc_layers: list[layers.Dense],
        dropout: float,
        optimizer: optimizers.Optimizer,
        loss: str,
        metrics: list[str]
    ):
        self.input = layers.Input(shape=in_size)
        x = initial_conv(self.input)
        
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = initial_pool(x)
        
        for block in ResBlocks:
            x = block(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        for layer in fc_layers:
            x = layer(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(dropout)(x)
        self.output = layers.Dense(out_shape, activation='sigmoid')(x)
        
        self.model = models.Model(inputs=self.input, outputs=self.output)
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            jit_compile=True
        )
        
        
        num_params = self.model.count_params()
        print('-'*40)
        print(f'Total number of parameters: {num_params}')
        print('-'*40)
    def fit(self, save_path = None, best_path = 'training/bestacc.txt', *args, **kwargs):
        checkpoint_callback = lambda: None
        with open(best_path, 'r') as f:    
            best_accuracy = float(f.read())
        if save_path is not None:
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                save_path,
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                initial_value_threshold=0#best_accuracy
            )

        
        hist = self.model.fit(*args, **kwargs, callbacks=[checkpoint_callback])
        
        best_accuracy_new = max(hist.history['val_accuracy'])
        if best_accuracy_new > best_accuracy:
            with open(best_path, 'w') as f:
                f.write(str(best_accuracy_new))
        return hist
        

    def save(self, path):
        self.sequential.save(path)
    def load(self, path):
        self.sequential = models.load_model(path)

