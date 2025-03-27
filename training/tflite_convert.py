import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot
import os


model = keras.models.load_model('model.h5')
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
    subset='both',
    interpolation='bilinear'
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
print
tflite_quant_model = converter.convert()
# Save the TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_quant_model)
#Test in Python
os.system('xxd -c 60 -i model.tflite > embedded/Proj1/include/model.h')
os.system("sed -i 's/unsigned char/const unsigned char/g' embedded/Proj1/include/model.h")
os.system("sed -i 's/const/alignas(8) const/g' embedded/Proj1/include/model.h")

# Initialize the TFLite interpreter
interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)

# Allocate the tensors
interpreter.allocate_tensors()

# Get input/output layer information
i_details = interpreter.get_input_details()[0]
o_details = interpreter.get_output_details()[0]

# Get input quantization parameters.
i_quant = i_details["quantization_parameters"]
i_scale      = i_quant['scales'][0]
i_zero_point = i_quant['zero_points'][0]

num_correct_samples = 0

val_rebatch = val.rebatch(1)
num_total_samples   = len(list(val_rebatch))

for i_value, o_value in val_rebatch:
  i_value = (i_value / i_scale) + i_zero_point
  i_value = tf.cast(i_value, dtype=tf.int8)
  interpreter.set_tensor(i_details["index"], i_value)
  interpreter.invoke()
  o_pred = interpreter.get_tensor(o_details["index"])[0]

  if (o_pred > 0) == o_value:
    num_correct_samples += 1

print("Accuracy:", num_correct_samples/num_total_samples)