from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Sequential

model = Sequential()

model.add(
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))
)

model.add(
    MaxPool2D(2, 2)
)

model.add(
    Conv2D(64, (3, 3), activation='relu')
)

model.add(
    MaxPool2D(2, 2)
)

model.add(
    Conv2D(128, (3, 3), activation='relu')
)

model.add(
    MaxPool2D(2, 2)
)

model.add(
    Flatten()
)

model.add(
    Dense(512, activation='relu')
)

model.add(
    Dense(4, activation='softmax')
)

print(model.summary())