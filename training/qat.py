import numpy as np
import tempfile
import tensorflow as tf
from tf_keras import models, optimizers, layers
import tf_keras as keras
import tensorflow_model_optimization as tfmot
from classifier import ResNet

image_size = (64, 64)




def remove_batchnorm(model: models.Model):
    inputs = model.input  # Model input tensor
    x = inputs
    
    layer_outputs = {}  # Track layer outputs for residual connections

    for i, layer in enumerate(model.layers):
        if isinstance(layer, layers.BatchNormalization):
            model.layers.remove(layer)
        



model = models.load_model('model.h5')
remove_batchnorm(model)

for layer in model.layers:
    print(layer.__class__.__name__)



print('Begin Quantization')
model = tfmot.quantization.keras.quantize_model(model)
print("End Quantization")
model.save('model_qat.h5')
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])


train, val = keras.utils.image_dataset_from_directory(
    directory = 'dataset',
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=16,
    image_size=image_size,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='both'
)
model.eval(val)
