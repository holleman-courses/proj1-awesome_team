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
    image_size=image_size,
    batch_size=None,
    shuffle=True,
    seed=42,
    validation_split=0.3,
    subset='both',
    interpolation='bilinear'
)



augmentation = models.Sequential([
    layers.RandomFlip(),
    layers.RandomRotation(0.4),
    #layers.RandomBrightness(0.2),
])
def augment(dataset: tf.data.Dataset, batch_size: int):
    autotune = tf.data.AUTOTUNE

    # Apply augmentation
    augmented_dataset = dataset.map(lambda x, y: (augmentation(x), y), num_parallel_calls=autotune)

    # Combine original and augmented datasets
    dataset = dataset.concatenate(augmented_dataset)

    # Shuffle, batch, and prefetch
    dataset = dataset.batch(batch_size, num_parallel_calls=autotune).prefetch(autotune)
    return dataset

train = augment(train, 64)

val = val.batch(360, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


model = models.load_model('model.h5')
print("Baseline Accuracy")
model.evaluate(val)
model.summary()

# QAT

qat_model = tfmot.quantization.keras.quantize_model(model)
qat_model.compile(
    optimizer=optimizers.Adam(lr=1e-3, weight_decay=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

hist = qat_model.fit(
    x=train,
    epochs=20,
    validation_data=val,
    callbacks=[gen_checkpoint_callback('qat_model.h5')]
)