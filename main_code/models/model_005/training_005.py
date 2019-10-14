import keras
from sklearn.model_selection import train_test_split
import numpy as np

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


print("Loading training and testing data from disk...")

# Loading in the data
X = np.load("../../../data/X.npy")
Y = np.load("../../../data/Y.npy")

# Splitting the data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/7)


print("Loading model from disk...")

# Loading the model
json_file = open("../../../data/models/model_002/model_005.json", "r")
loaded_json_file = json_file.read()
json_file.close()

with keras.utils.CustomObjectScope({'GlorotUniform': keras.initializers.glorot_uniform()}):
    model = keras.models.model_from_json(loaded_json_file)

model.load_weights("../../../data/models/model_005/model_005.h5")


print("Training...")

# Configure the training details, e.g. what optimiser to use
model.compile(loss = keras.losses.categorical_crossentropy,
             optimizer = keras.optimizers.Adadelta(),
             metrics = ['accuracy'])

# Fit the model. The loss and accuracy will be outputed by default.
history = model.fit(X_train, Y_train,
          batch_size = 52,
          epochs = 100)

# Evaluate the performance
performance = model.evaluate(X_test, Y_test)
print('The loss is %.3f and the accuracy is %.3f on the test data' 
       % tuple(performance))


print("Saving the 005 model...")

# Serialise model to JSON
model_json = model.to_json()
with open("../../../data/models/model_005/model_005.json", "w") as json_file:
    json_file.write(model_json)

# Serialise weights to HDF5
model.save_weights("../../../data/models/model_005/model_005.h5")