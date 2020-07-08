from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from preprocessing import test_generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

model = load_model('models/model.h5')

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

TEST_IMAGES_DIR = os.path.join(BASE_DIR, 'test-images')

test_imagegen = ImageDataGenerator(
    rescale=1. / 255
)
test_images_generator = test_imagegen.flow_from_directory(
    TEST_IMAGES_DIR,
    target_size=(150, 150),
    # batch_size=32,
    class_mode='binary'
)
# img = image.load_img('cars_test/L/00178.jpg', target_size=(150, 150))

# img = image.img_to_array(img)

# img = np.expand_dims(img, axis=0)
# print(img.shape)
# img = np.reshape(img, (1,  4 * 4 * 512))
predictions = model.predict(test_images_generator)
#
for prediction in predictions:
    # print(prediction)
    if np.argmax(prediction) == 0:
        print('L GH 1.50 ', prediction[0])
    elif np.argmax(prediction) == 1:
        print('M GH 1.00 ', prediction[1])
    elif np.argmax(prediction) == 2:
        print('P GH 2.00 ', prediction[2])
    else:
        print('S GH 0.5 ', prediction[3])
    # print(np.argmax(prediction))
# print(model.evaluate(test_generator))
# print(prediction)
# if prediction[0][] == 1.0:
#     print('L GH 1.50')
# elif prediction[1] == 1.0:
#     print('M GH 1.00')
# elif prediction[2] == 1.0:
#     print('P GH 2.00')
# elif prediction[3] == 1.0:
#     print('S GH 0.50 ')