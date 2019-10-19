import keras 
from sklearn.model_selection import train_test_split
import numpy as np
import keras.backend as K

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import sys
sys.path.append('../../')
import metrics

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


print("Loading training and testing data from disk...")

# Loading in the data
X = np.load("../../../data/X.npy")
y = np.load("../../../data/Y.npy")

#-------------------------------------DONT EDIT ABOVE LINE-------------------------------------------------

# Splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/7)

def test_model(num):
    print("Loading model from disk...")

    # Loading the model
    json_file = open("../../../data/models/model_00{}/model_00{}_trained.json".format(num, num), "r")
    loaded_json_file = json_file.read()
    json_file.close()

    with keras.utils.CustomObjectScope({'GlorotUniform': keras.initializers.glorot_uniform()}):
        model = keras.models.model_from_json(loaded_json_file)

    model.load_weights("../../../data/models/model_00{}/model_00{}_trained.h5".format(num, num))

    y_pred = model.predict(X_test)
    print("Prediction:\n", y_pred)
    print("Ground Truth:\nHas {}/{} FALSE labels\n".
            format(np.count_nonzero(y_test[:, 0] == 1), y_test.shape[0]),
            y_test)
    metrics.confusion_matrix(y_pred, y_test) # Producing a confusion matrix
    metrics.ROC(y_pred, y_test, 1)

#for num in range(1, 6):
#    test_model(num)

test_model(5)
