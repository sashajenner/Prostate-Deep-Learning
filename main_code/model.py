    ## Importing necessary libraries
import tensorflow.keras as keras # Importing required neural network module
import csv
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('../data/info.csv') as csv_file:
    reader = csv.reader(csv_file)
    lines = list(reader)

lines.pop(0) # Removing the header

print("Preparing images and labels...")

# Declaring empty lists to hold the image and label data
t2w_images = []
t2w_labels = []
index = 0
for index in range(len(lines)):
    entry = lines[index]

    # Extracting important fields
    patient_id = entry[1]
    fid = int(entry[3])
    label = True if entry[-1] == "TRUE" else False

    # Loading each t2w image and locating the required slice
    t2w_cropped_file = nib.load('../data/mri/{}/t2w_cropped_0{}.nii'.format(patient_id, fid))
    t2w_cropped_image = t2w_cropped_file.get_fdata()
    
    t2w_images.append(t2w_cropped_image)
    t2w_labels.append(label)

# Turning the image and label lists to a numpy arrays
t2w_images = np.array(t2w_images)
t2w_labels = np.array(t2w_labels)

# Image dimensions
num_images, img_rows, img_cols = t2w_images.shape # (326, 147, 147)

# Since the images are greyscale - 1
t2w_images = t2w_images.reshape(num_images, img_rows, img_cols, 1)

print('Images shape:', t2w_images.shape) # Testing

# Normalise the data (so that they lie between 0 and 1)
t2w_images = t2w_images.astype('float32') / 255.0

# Preprocessing the label data
t2w_labels = keras.utils.to_categorical(t2w_labels, 2)

# Splitting the data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(t2w_images, t2w_labels, test_size = 1/7)

    ## Define the deep learning structure

print("Building model...")

# Initialise the model, it's empty at the beginning
model = keras.models.Sequential()

# Add the first convolutional layer
model.add(keras.layers.Conv2D(32, kernel_size = (3,3), 
                              activation='relu', input_shape = (img_rows, img_cols, 1)))
# Add the first pooling layer
model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))

# Similarly the second convolutional layer
model.add(keras.layers.Conv2D(64, (3,3), activation = 'relu'))
# The second pooling layer
model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))

# Flatten the image (pulled into an array)
model.add(keras.layers.Flatten())
# Feed the array into a densely-connected neural network
model.add(keras.layers.Dense(128, activation = 'relu'))
# Use the softmax to map the output to probabilities
model.add(keras.layers.Dense(2, activation = 'softmax'))

print("Training...")

# Configure the training details, e.g. what optimiser to use
model.compile(loss = keras.losses.categorical_crossentropy,
             optimizer = keras.optimizers.Adadelta(),
             metrics = ['accuracy'])

# Fit the model. The loss and accuracy will be outputed by default.
history = model.fit(X_train, Y_train,
          batch_size = 64,
          epochs = 20)

# Evaluate the performance
performance = model.evaluate(X_test, Y_test)
print('The loss is %.3f and the accuracy is %.3f on the test data' 
       % tuple(performance))
