import numpy as np
import tempfile
import tensorflow as tf
from tf_keras import models, optimizers, layers
import tf_keras as keras
import tensorflow_model_optimization as tfmot
from classifier import ResNet

image_size = (64, 64)


model = models.load_model('model.h5')

print('Begin Quantization')
model = tfmot.quantization.keras.quantize_model(model)
print("End Quantization")

# for layer in model.layers:
#     print(layer.name)

model.save('model_qat.h5')
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])


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

print("Before tuning")
model.evaluate(val)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    'model_qat_tuned.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    initial_value_threshold=0#best_accuracy
)

#hist = model.fit(train, epochs=100, validation_data=val, validation_batch_size=179, callbacks=[checkpoint_callback])
#print(f"Best Validation Accuracy: {max(hist.history['val_accuracy'])}")

print("After tuning")
with tfmot.quantization.keras.quantize_scope():
  loaded_model = keras.models.load_model('model_qat_tuned.h5')
loaded_model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

loaded_model.evaluate(val)
