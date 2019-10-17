import tensorflow.keras as keras
import numpy as np


# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


print("Loading X data from disk...")

# Loading in the data
X = np.load("../../../data/X.npy")

# Image dimensions
num_images, img_rows, img_cols, channels = X.shape # (3586, 147, 147, 3)

#-------------------------------------DONT EDIT ABOVE LINE-------------------------------------------------

    ## Define the deep learning structure

print("Building model...")

# Initialise the model, it's empty at the beginning
model = keras.models.Sequential()

# Add the first convolutional layer
# SeparationalConv2D?
model.add(keras.layers.Conv2D(64, kernel_size = (3,3), 
                              activation='relu', input_shape = (img_rows, img_cols, channels)))
# Add the first pooling layer
model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))

# Similarly the second convolutional layer
model.add(keras.layers.Conv2D(32, (3,3), activation = 'relu'))
# The second pooling layer
model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))

# Flatten the image (pulled into an array)
model.add(keras.layers.Flatten())
# Feed the array into a densely-connected neural network
model.add(keras.layers.Dense(128, activation = 'relu'))
# Use the softmax to map the output to probabilities
model.add(keras.layers.Dense(2, activation = 'softmax'))

model.summary()

print("Saving model 003 to disk...")

# Serialise model to JSON
model_json = model.to_json()
with open("../../../data/models/model_003/model_003_init.json", "w") as json_file:
    json_file.write(model_json)

# Serialise weights to HDF5
model.save_weights("../../../data/models/model_003/model_003_init.h5")


# Writing the model summary to a file
import sys
sys.stdout = open("../../../data/models/model_003/model_003_summary.txt", "w")
model.summary()
