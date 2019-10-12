    ## Importing necessary libraries
import keras # Importing required neural network module
import cv2
import csv
import glob
import numpy as np
import nibabel as nib

with open('../data/info.csv') as csv_file:
    reader = csv.reader(csv_file)
    lines = list(reader)

lines.pop(0) # Removing the header

X_data = []
for entry in lines:
    # Extracting important fields
    patient_id = entry[1]
    fid = int(entry[3])

    # Loading each t2w image and locating the required slice
    t2w_cropped_file = nib.load('../data/mri/{}/t2w_cropped_0{}.nii'.format(patient_id, fid))
    t2w_cropped_image = t2w_cropped_file.get_fdata()
    
    X_data.append(t2w_cropped_image)

print('X_data shape:', np.array(X_data).shape)

# Image dimensions
img_rows, img_cols = 147, 147


    ## Define the deep learning structure

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
model.add(keras.layers.Dense(10, activation = 'softmax'))

# Configure the training details, e.g. what optimiser to use
model.compile(loss = keras.losses.categorical_crossentropy,
             optimizer = keras.optimizers.Adadelta(),
             metrics = ['accuracy'])

# Fit the model. The loss and accuracy will be outputed by default.
history = model.fit(train_images, train_labels,
          batch_size = 64,
          epochs = 20)

# Evaluate the performance
performance = model.evaluate(test_images, test_labels)
print('The loss is %.3f and the accuracy is %.3f on the test data' 
       % tuple(performance))
