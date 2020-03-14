from model_architecture import model
from preprocessing import train_generator, test_generator, validation_generator
from load_data import TRAIN_DIR, TEST_DIR, VALIDATION_DIR
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import models

# model = load_model('models/model.h5')
history = model.fit_generator(
    train_generator,
    epochs=5,
    validation_data=validation_generator
)

model.save('models/model.h5')
