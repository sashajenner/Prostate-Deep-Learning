import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

input_shape = (28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = keras.models.Sequential() # initialise model
model.add(keras.layers.Conv2D(32, kernel_size(3, 3), activation='relu', input_shape = input_shape))
# 32 filters using kernel size 3 x 3
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(loss =keras.losses.categorical_crossentropy...)
history = model.fit(x_train, y_train, batch_size = 128, epochs = 10)


