from model_architecture import model
from preprocessing import train_generator, test_generator, validation_generator
from load_data import TRAIN_DIR, TEST_DIR, VALIDATION_DIR
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import models
conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 20


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))

    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )

    i = 0
    for inputs_batch, labels_batch in generator:

        features_batch = conv_base.predict(inputs_batch)

        features[i * batch_size: (i + 1) * batch_size] = features_batch

        labels[i * batch_size: (i + 1) * batch_size] = labels_batch

        i += 1

        if i * batch_size >= sample_count:
            break

        return features, labels


train_features, train_labels = extract_features(TRAIN_DIR, 2000)

validation_features, validation_labels = extract_features(VALIDATION_DIR, 1000)

test_features, test_labels = extract_features(TEST_DIR, 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))


history = model.fit(
    train_features,
    train_labels,
    epochs=30,
    validation_data=(validation_features, validation_labels)
)

model.save('models/model.h5')
