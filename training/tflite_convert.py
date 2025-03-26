import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot
import os


model = keras.models.load_model('qat_model.h5')
print(model.summary())
image_size = (176, 144)

_, val = keras.utils.image_dataset_from_directory(
    directory = 'dataset',
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=1,
    image_size=image_size,
    shuffle=True,
    seed=42,
    validation_split=0.3,
    subset='both'
)

def representative_data_gen():
  for input_value, _ in val:
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8  

tflite_quant_model = converter.convert()
# Save the TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_quant_model)
#Test in Python
os.system('xxd -i model.tflite > embedded/Proj1/include/model.h')