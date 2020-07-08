from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

model = Sequential()

# base_model = VGG16(
#     weights='imagenet',
#     include_top=False
# )
# model.add(
#     base_model
# )

model.add(Conv2D(32, (3, 3), activation='relu',
                 input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
# model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
# model.add(
#     Dropout(0.5)
# )

model.add(
    Dense(
        4,
        activation='softmax'
    )
)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
