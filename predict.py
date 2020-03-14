from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from preprocessing import test_generator
model = load_model('models/model.h5')

img = image.load_img('cars_validation/M/00007.jpg', target_size=(150, 150))

img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)
# print(img.shape)
# img = np.reshape(img, (1,  4 * 4 * 512))
prediction = model.predict(img)[0]
#
# print(model.evaluate(test_generator))
print(prediction)
if prediction[0] == 1.0:
    print('E GH 0.50')
elif prediction[1] == 1.0:
    print('L GH 1.00')
elif prediction[2] == 1.0:
    print('M GH 1.00')
elif prediction[3] == 1.0:
    print('S GH 0.50 ')