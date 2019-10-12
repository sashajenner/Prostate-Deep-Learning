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

! mkdir cropped_images

#image cropping

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import nibabel

for i in range(1, len(content)):
  t2w_image = nibabel.load('data/mri/%7s/t2w.nii'%content[i][1])
  img = t2w_image.get_fdata()
  mask = img.copy()
  dim_list = content[i][4].split(" ")
  #x_coord = int(dim_list[0])
  #y_coord = int(dim_list[1])
  slice_height = int(dim_list[2])
  cropped_img = mask[int(t2w_image.shape[0]/2)-120:int(t2w_image.shape[0]/2) + 120, int(t2w_image.shape[1]/2)-120:int(t2w_image.shape[1]/2)+120, slice_height]
  plt.imsave("cropped_images/cropped_%7s"%content[i][1], cropped_img)

#! ls cropped_images

img = mpimg.imread("cropped_images/cropped_pt_0089.png")
plt.imshow(img)

