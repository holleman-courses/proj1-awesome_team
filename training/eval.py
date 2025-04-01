from tf_keras import models
import tf_keras as keras
from tf_keras import layers
import tensorflow as tf

model: models.Model = models.load_model('model.h5')

params = model.count_params()
print('-'*40)
print(f'Total number of parameters: {params}')
print('-'*40)


model.compile(metrics = [
    keras.metrics.BinaryAccuracy(),
    keras.metrics.Precision(),
    keras.metrics.Recall(),
    keras.metrics.TruePositives(),
    keras.metrics.TrueNegatives(),
    keras.metrics.FalsePositives(),
    keras.metrics.FalseNegatives(),
])

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
    #aug2 = dataset.map(lambda x, y: (augmentation(x), y), num_parallel_calls=autotune)

    # Combine original and augmented datasets
    dataset = dataset.concatenate(augmented_dataset)#.concatenate(aug2)

    # Shuffle, batch, and prefetch
    dataset = dataset.batch(batch_size, num_parallel_calls=autotune).prefetch(autotune)
    return dataset

train = augment(train, 64)

val = val.batch(360, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


print('-'*40)
print('Training')
print('-'*40)
model.evaluate(train)

print('-'*40)
print('Validation')
print('-'*40)
model.evaluate(val)
