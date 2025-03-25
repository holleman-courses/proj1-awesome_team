import tensorflow as tf
from tf_keras import layers, optimizers, models
import tf_keras as keras


def build_res_block(filters, kernel_size=3, strides=(1,1), padding='same', activation='relu', dropout=0):
    def res_block(x, use_batch_norm=True):
        shortcut = x
        
        out = layers.SeparableConv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        if use_batch_norm:
            out = layers.BatchNormalization()(out)
        out = layers.Activation(activation)(out)
        
        out = layers.SeparableConv2D(filters, kernel_size, strides=(1,1), padding=padding, use_bias=False)(out)
        if use_batch_norm:
            out = layers.BatchNormalization()(out)
        out = layers.Activation(activation)(out)
        
        if strides != (1,1) or x.shape[-1] != filters:
            shortcut = layers.SeparableConv2D(filters, kernel_size=1, strides=strides, padding='same', use_bias=False)(x)
            if use_batch_norm:
                shortcut = layers.BatchNormalization()(shortcut)
        
        out = layers.Add()([out, shortcut])
        out = layers.Dropout(dropout)(out)
        return out
    return res_block

class ResNet:
    
    def gen_model(self, use_batch_norm=True):
        self.input = layers.Input(shape=self.in_size)
        x = self.initial_conv(self.input)
        
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = self.initial_pool(x)
        
        for block in self.ResBlocks:
            x = block(x, use_batch_norm=use_batch_norm)
        
        x = layers.GlobalAveragePooling2D()(x)
        for layer in self.fc_layers:
            x = layer(x)
            if use_batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(self.dropout)(x)
        self.output = layers.Dense(self.out_shape, activation='sigmoid')(x)
        
        self.model = models.Model(inputs=self.input, outputs=self.output)
        
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
        self.initial_conv = initial_conv
        self.initial_pool = initial_pool
        self.ResBlocks = ResBlocks
        self.fc_layers = fc_layers
        self.dropout = dropout
        
        
        
        
        
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

