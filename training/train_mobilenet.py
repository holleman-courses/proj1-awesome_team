import tf_keras as keras
import tensorflow as tf
from tf_keras import layers, optimizers, models
# import mobilenet v3
from tf_keras.applications import MobileNetV3Small

image_size = (48, 48)
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

p_drop = 0.4
mobile_net = MobileNetV3Small(
    input_shape= (*image_size, 1),
    include_top=False,
    weights='imagenet',
    #alpha=0.4,
)
mobile_net.trainable = False

model = models.Sequential([
    mobile_net,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(p_drop),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer = optimizers.Adam(learning_rate=5e-3, weight_decay=1e-5),
    loss = keras.losses.BinaryCrossentropy(from_logits=False),
    metrics = ['accuracy']
)

print('-'*50)
print('Total number of parameters:', model.count_params())
print('-'*50)


checkpoint_callback = keras.callbacks.ModelCheckpoint(
    'models/mobilenet.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    initial_value_threshold=0
)
model.fit(
    train,
    validation_data=val,
    epochs=50,
    callbacks=[checkpoint_callback]
)
