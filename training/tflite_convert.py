import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot

with tfmot.quantization.keras.quantize_scope():
  model = keras.models.load_model('model_qat_tuned.h5')

image_size = (64, 64)

_, val = keras.utils.image_dataset_from_directory(
    directory = 'dataset',
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=1,
    image_size=image_size,
    shuffle=True,
    seed=42,
    validation_split=0.2,
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

# tflite_quant_model = converter.convert()
# # Save the TFLite model
# with open("model.tflite", "wb") as f:
#     f.write(tflite_quant_model)
# Test in Python


interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get quantization parameters
input_scale = input_details[0]['quantization'][0]
input_zero_point = input_details[0]['quantization'][1]

print(f"Input Scale: {input_scale} | Input Zero Point: {input_zero_point}")
print(f"Output Scale: {output_details[0]['quantization'][0]} | Output Zero Point: {output_details[0]['quantization'][1]}")

size = 0
for tensor in interpreter.get_tensor_details():
    match tensor['dtype']:
        case np.int8:
            size += np.prod(tensor['shape'])
        case np.float32:
            size += np.prod(tensor['shape']) * 4
        case np.int32:
            if 'pseudo' in tensor['name']:
                continue
            size += np.prod(tensor['shape']) * 4
        case _:
            raise ValueError(f"Unsupported data type: {tensor['dtype']}")
    
            

print(f"Arena Size: {size} bytes")