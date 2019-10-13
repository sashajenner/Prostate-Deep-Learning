    ## Importing necessary libraries
import tensorflow.keras as keras # Importing required neural network module
import csv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt # Testing


# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('../data/info.csv') as csv_file:
    reader = csv.reader(csv_file)
    lines = list(reader)

lines.pop(0) # Removing the header

print("Preparing images and labels...")

# Declaring empty lists to hold the image and label data
X = []
Y = []
index = 0
for index in range(len(lines)):
    entry = lines[index]

    # Extracting important fields
    patient_id = entry[1]
    fid = int(entry[3])
    label = True if entry[-1] == "TRUE" else False

    # Loading each cropped image
    t2w_cropped_file = nib.load('../data/mri/{}/t2w_cropped_0{}.nii'.format(patient_id, fid))
    t2w_cropped_image = t2w_cropped_file.get_fdata()
    ktrans_cropped_file = nib.load('../data/mri/{}/ktrans_cropped_0{}.nii'.format(patient_id, fid))
    ktrans_cropped_image = ktrans_cropped_file.get_fdata()
    adc_cropped_file = nib.load('../data/mri/{}/adc_cropped_0{}.nii'.format(patient_id, fid))
    adc_cropped_image = adc_cropped_file.get_fdata()

    merged_crop = np.dstack((t2w_cropped_image, ktrans_cropped_image, adc_cropped_image))
    
    # Testing
    #plt.imshow(merged_crop)
    #plt.show()

    X.append(merged_crop)
    Y.append(label)

# Turning the image and label lists to a numpy arrays
X = np.array(X)
Y = np.array(Y)

print('Images shape:', X.shape) # Testing

# Normalise the data (so that they lie between 0 and 1)
X = X.astype('float32') / 255.0

# Preprocessing the label data
Y = keras.utils.to_categorical(Y, 2)


print("Saving the data...")

    ## Saving the data
np.save("../data/X.npy", X)
np.save("../data/Y.npy", Y)
