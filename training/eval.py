from tf_keras import models
import tf_keras as keras

image_size = (176, 144)

model: models.Model = models.load_model('model.h5')
model.compile(metrics = [
    keras.metrics.BinaryAccuracy(),
    keras.metrics.Precision(),
    keras.metrics.Recall(),
    keras.metrics.TruePositives(),
    keras.metrics.TrueNegatives(),
    keras.metrics.FalsePositives(),
    keras.metrics.FalseNegatives(),
])


_, val = keras.utils.image_dataset_from_directory(
    directory = 'dataset',
    labels='inferred',
    label_mode='binary',
    color_mode='grayscale',
    batch_size=32,
    image_size=image_size,
    shuffle=True,
    seed=42,
    validation_split=0.3,
    subset='both'
)

model.evaluate(val)
