import tensorflow as tf
from tf_keras import layers, optimizers, models
import tf_keras as keras


def build_res_block(filters, kernel_size=3, strides=(1,1), padding='same', activation='relu', dropout=0):
    def res_block(x):
        shortcut = x
        
        out = layers.BatchNormalization()(x)
        out = layers.Activation(activation)(out)
        out = layers.SeparableConv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        
        out = layers.BatchNormalization()(out)
        out = layers.Dropout(dropout)(out) # Dropout at P5
        out = layers.Activation(activation)(out)
        out = layers.SeparableConv2D(filters, kernel_size, strides=(1,1), padding=padding, use_bias=False)(out)
        
        if strides != (1,1) or x.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, kernel_size=1, strides=strides, padding='same', use_bias=False)(x)
        
        out = layers.Add()([out, shortcut])
        
        return out
    return res_block

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
        
        
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout)(x) # Dropout at H4
        x = layers.GlobalAveragePooling2D()(x)
        
        for layer in fc_layers:
           x = layer(x)
           x = layers.BatchNormalization()(x)
           x = layers.Activation('relu')(x)
            
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
    def fit(
        self,
        save_path = None,
        best_path = 'training/bestacc.txt',
        reset_best_acc = False,
        stop_patience = 10,
        *args, **kwargs):
        checkpoint_callback = lambda: None
        if reset_best_acc:
            best_accuracy = 0
        else:
            with open(best_path, 'r') as f:    
                best_accuracy = float(f.read())
        if save_path is not None:
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                save_path,
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                initial_value_threshold=best_accuracy
            )
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=stop_patience,
            mode='max'
        )

        
        hist = self.model.fit(*args, **kwargs, callbacks=[checkpoint_callback, early_stop])
        
        best_accuracy_new = max(hist.history['val_accuracy'])
        print(f"Best Validation Accuracy: {best_accuracy_new}")
        if best_accuracy_new > best_accuracy:
            with open(best_path, 'w') as f:
                f.write(str(best_accuracy_new))
        return hist
        
    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)
    def save(self, path):
        self.model.save(path)
    def load(self, path):
        self.model = models.load_model(path)

