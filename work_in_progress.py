! wget "https://www.dropbox.com/s/v56dirv1z14fy4j/dalyell_2019_s2_train.zip?dl=0#" -O data.zip
! ls -la
! mkdir data
! unzip data.zip -d data/
! ls data
! ls data/mri
! ls data/mri/pt_0000/
import csv

with open('data/info.csv') as f:

  reader = csv.reader(f)

  content = list(reader)

  
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import nibabel

! cd cropped_images
! mkdir x_train
! mkdir y_train
! mkdir x_test
! mkdir y_test

#cropping and organising for generator

for i in range(1, len(content)):
  t2w_image = nibabel.load('data/mri/%7s/t2w.nii'%content[i][1])
  img = t2w_image.get_fdata()
  mask = img.copy()
  dim_list = content[i][4].split(" ")
  #x_coord = int(dim_list[0])
  #y_coord = int(dim_list[1])
  slice_height = int(dim_list[2])
  cropped_img = mask[int(t2w_image.shape[0]/2)-120:int(t2w_image.shape[0]/2) + 120, int(t2w_image.shape[1]/2)-120:int(t2w_image.shape[1]/2)+120, slice_height]
  if(i < len(content)//3):
    plt.imsave("x_train/cropped_%7s"%content[i][1], cropped_img)
    plt.imsave("x_test/cropped_%7s"%content[i][1], cropped_img)
  else:
    plt.imsave("y_train/cropped_%7s"%content[i][1], cropped_img)
    plt.imsave("y_test/cropped_%7s"%content[i][1], cropped_img)
    
from keras.preprocessing.image import ImageDataGenerator

path = "cropped_images"

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

gen = datagen.flow_from_directory(path, target_size=(140,140), color_mode="rgba", batch_size=32)

#custom metrics i.e. specificity and sensitivity

import keras.backend as K

​

#custom specificty/sensitivity due to keras not having these

def specificity(x_pred, x_true):

  """
  specificity:
  x_pred = matrix of labels that are predicted to be true
  x_true = labels that are true by the GT
  not_true = labels that are not true by the GT
  not_pred = labels that are predicted to be not true
  return spec
  """

  not_true = 1-x_true

  not_pred = 1-x_pred

  FP = K.sum(not_true * x_pred)

  TN = K.sum(not_true * not_pred)

  spec = TN/(TN+FP)

  return spec


def sensitivity(x_pred, x_true):

  """
  sensitivity:
  x_pred = matrix of labels that are predicted to be true
  x_true = labels that are true by the GT
  not_true = labels that are not true by the GT
  not_pred = labels that are predicted to be not true
  return sens
  """

  # subtracting the tensors x_true and x_pred from 1 should give their complement tensors

  # since this is a binary classification

  not_true = 1-x_true

  not_pred = 1-x_pred

  FN = K.sum(not_true * x_pred)

  TP = K.sum(x_true * x_pred)

  sens = TP/(TP + FN)

  return sens

#the model

import keras
import sklearn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split


model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(2,2), activation="relu", input_shape=(140,140,1)))
 
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(32, (2,2), activation="relu"))

model.add(keras.layers.MaxPooling2D((2,2)))

model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adadelta(),
             metrics=['accuracy',specificity,sensitivity])
  
result = model.fit_generator(gen, 
                             steps_per_epoch=len(content)//32, 
                             epochs=100,
                             verbose=1)
