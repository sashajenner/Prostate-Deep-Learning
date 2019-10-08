# Define the deep learning structure

# Initialise the model, it's empty at the beginning
model = keras.models.Sequential()


model.add(keras.layers.Conv2D(32, kernel_size=(3,3), 
                              activation='relu', input_shape=input_shape))

model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))

model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(64, activation='relu'))

model.add(keras.layers.Dense(10, activation='softmax'))

model.add(keras.layers.Dense(128, activation='relu'))

model.add(keras.layers.Dense(10, activation='softmax'))
