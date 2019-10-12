
'''
SCDL1991 Dalyell Group Week 4 Exercise
All completed by Sasha Jenner
'''

##########################################################################################################

import skimage, numpy as np

#1a
koalaImage = skimage.io.imread('koala.tiff')

#1b
p75Koala = np.percentile(koalaImage, 75) #153.0

#1c
koalaProcessed = np.array(koalaImage, copy = True)
koalaProcessed[koalaProcessed > p75Koala] = 0.5 * koalaProcessed[koalaProcessed > p75Koala] 

#1d
skimage.io.imsave('koala_processed.tiff', koalaProcessed)

#1e
koalaDiff = koalaImage - koalaProcessed
skimage.io.imsave('koala_diff.tiff', koalaDiff)

#1f
koalaDiffRotate = np.rot90(koalaDiff, -1)
inputShape = koalaDiffRotate.shape
import cv2
koalaDiffRotateScale = cv2.resize(koalaDiffRotate, dsize = (inputShape[1] // 2, inputShape[0] // 2), interpolation = cv2.INTER_NEAREST)
skimage.io.imsave('koala_diff_rot.tiff', koalaDiffRotateScale)

##########################################################################################################

#2a
import pydicom
dcm = pydicom.dcmread('MR000008')

#2b
print(dcm) 
'''
prints out the meta-data of the file on separate lines in the form
(<num1>, <num2>) [<headerName>]                    <twoCharsIdentifier>: <information>
'''

#2c
data = dcm.pixel_array

#2d
import matplotlib, matplotlib.pyplot as plt
plt.imshow(data)
plt.show()

#2e
matplotlib.image.imsave('MR000008.tiff', data)

#2f
p75Data = np.percentile(data, 75) #208.0
#i
dataProcessed = np.array(data, copy = True)
dataProcessed[dataProcessed > p75Data] = 0.5 * dataProcessed[dataProcessed > p75Data] 
matplotlib.image.imsave('MR000008_processed.tiff', dataProcessed)
#ii
dataDiff = data - dataProcessed
matplotlib.image.imsave('MR000008_diff.tiff', dataDiff)
#iii
dataDiffRotate = np.rot90(dataDiff, -1)
dataDiffRotateScale = cv2.resize(dataDiffRotate, dsize = (dataDiffRotate.shape[1] // 2, dataDiffRotate.shape[0] // 2), interpolation = cv2.INTER_NEAREST)
matplotlib.image.imsave('MR000008_diff_rot.tiff', dataDiffRotateScale)

##########################################################################################################

#3
import zipfile
with zipfile.ZipFile('SE000001.zip', 'r') as zip_ref:
    zip_ref.extractall('./') #Extract contents of zip file to home directory

for i in range(0, 3): 
    #Plotting and saving MR00000[0-2]
    filepath = 'SE000001/MR00000{}'.format(i)
    dcmSlice = pydicom.dcmread(filepath)
    dataSlice = dcmSlice.pixel_array
    plt.imshow(dataSlice)
    plt.show()
    matplotlib.image.imsave(filepath + '.tiff', dataSlice)
    
    #Finding the 75 percentile SI for each
    print("MR00000{}: 75th percentile SI is {}".format(i, np.percentile(dataSlice, 75)))
