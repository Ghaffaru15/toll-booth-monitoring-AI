import cv2
from tensorflow.keras.models import load_model
from datetime import datetime
import numpy as np

from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
model = load_model('models/model.h5')
cap = cv2.VideoCapture(0)
camera_height = 500


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

TEST_IMAGES_DIR = os.path.join(BASE_DIR, 'image_capture')

test_imagegen = ImageDataGenerator(
    rescale=1. / 255
)
test_images_generator = test_imagegen.flow_from_directory(
    TEST_IMAGES_DIR,
    target_size=(150, 150),
    # batch_size=32,
    class_mode='binary'
)

# DB Connection
import sqlite3

conn = sqlite3.connect('store.db')
# print('connected')
while True:
    check, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # flip the frame
    frame = cv2.flip(frame, 1)

    # rescaling camera output
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * camera_height)  # landscape orientation - wide range
    frame = cv2.resize(frame, (res, camera_height))

    # add rectangle
    cv2.rectangle(frame, (300, 75), (650, 425), (240, 100, 0), 2)

    # get roi
    roi = frame[75 + 2:425 - 2, 300 + 2: 650 - 2]

    # parse bgr to rgb
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # resize to 224 224
    roi = cv2.resize(roi, (150, 150))
    cv2.imwrite('image_capture/images/car.jpg', roi)

    prediction = model.predict(test_images_generator)

    if np.argmax(prediction[0]) == 0 and prediction[0][0] > 0.8:

        print('L GH 1.50 ', prediction[0][0])

        conn.execute("INSERT INTO predictions (car_type,price,date) \
              VALUES ('L', 1.50, '" + str(datetime.now()) + "')")

        conn.commit()

    elif np.argmax(prediction[0]) == 1 and prediction[0][0] > 0.8:

        print('M GH 1.00 ', prediction[0][1])

        conn.execute("INSERT INTO predictions (car_type,price,date) \
                     VALUES ('M', 1.00, '" + str(datetime.now()) + "')")

        conn.commit()

    elif np.argmax(prediction[0]) == 2 and prediction[0][0] > 0.8:

        print('P GH 2.00 ', prediction[0][2])

        conn.execute("INSERT INTO predictions (car_type,price,date) \
                     VALUES ('P', 2.00, '" + str(datetime.now()) + "')")

        conn.commit()

    elif np.argmax(prediction[0]) == 3 and prediction[0][0] > 0.8:

        print('S GH 0.5 ', prediction[0][3])

        conn.execute("INSERT INTO predictions (car_type,price,date) \
                     VALUES ('S', 0.50, '" + str(datetime.now()) + "')")

        conn.commit()

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

