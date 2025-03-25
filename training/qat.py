import numpy as np
import tempfile
import tensorflow as tf
from tf_keras import models, optimizers, layers
import tf_keras as keras
import tensorflow_model_optimization as tfmot
from classifier import ResNet


def gen_checkpoint_callback(path):
  return keras.callbacks.ModelCheckpoint(
      path,
      save_best_only=True,
      monitor='val_accuracy',
      mode='max',
  )

image_size = (176, 144)

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


model = models.load_model('model.h5')
print("Baseline Accuracy")
model.evaluate(val)

print('Begin Pruning')
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)
}

prune_callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    gen_checkpoint_callback('models/pruned_model.h5')
]

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
opt = optimizers.Adam(learning_rate=1e-5, weight_decay=1e-7)
pruned_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

print("Pruning Tuning")
pruned_model.fit(train, epochs=20, validation_data=val, validation_batch_size=179, callbacks=prune_callbacks)

del pruned_model

pruned_model = models.load_model('models/pruned_model.h5')
print("Pruned Model Accuracy")
pruned_model.evaluate(val)

stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)


print("Begin Clustering")


from tensorflow_model_optimization.python.core.clustering.keras.experimental import (
    cluster,
)

cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

cluster_weights = cluster.cluster_weights

clustering_params = {
  'number_of_clusters': 8,
  'cluster_centroids_init': CentroidInitialization.KMEANS_PLUS_PLUS,
  'preserve_sparsity': True
}

sparsity_clustered_model = cluster_weights(stripped_pruned_model, **clustering_params)

sparsity_clustered_model.compile(optimizer=optimizers.Adam(learning_rate=1e-5, weight_decay=1e-7),
              loss=keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

print('Train sparsity preserving clustering model:')
sparsity_clustered_model.fit(train, epochs=20, validation_data=val, validation_batch_size=179, callbacks=[gen_checkpoint_callback('models/sparsity_clustered_model.h5')])
del sparsity_clustered_model
sparsity_clustered_model = models.load_model('models/sparsity_clustered_model.h5')

stripped_clustered_model = tfmot.clustering.keras.strip_clustering(sparsity_clustered_model)

print("Begin PCQuantization Aware Training")
# PCQAT
quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(
              stripped_clustered_model)
pcqat_model = tfmot.quantization.keras.quantize_apply(
              quant_aware_annotate_model,
              tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(preserve_sparsity=True))

pcqat_model.compile(optimizer=optimizers.Adam(learning_rate=1e-5, weight_decay=1e-7),
              loss=keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])
print('Train pcqat model:')
pcqat_model.fit(train, epochs=20, validation_data=val, validation_batch_size=179, callbacks=[gen_checkpoint_callback('models/pcqat_model.h5')])
del pcqat_model
pcqat_model = models.load_model('models/pcqat_model.h5')
print("PCQAT Model Accuracy")
pcqat_model.evaluate(val)

print("Begin TFLite Conversion")

def representative_data_gen():
  for input_value, _ in val:
    yield [input_value]
  
converter = tf.lite.TFLiteConverter.from_keras_model(pcqat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
# Save the TFLite model
with open("models/model.tflite", "wb") as f:
    f.write(tflite_quant_model)


