import csv
import nibabel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.io

with open('info.csv') as f:
    reader = csv.reader(f)
    lines = list(reader)

for i in range(10):
    print("%15s" * 7 % tuple(lines[i]))

t2w = nibabel.load('mri/pt_0000/t2w.nii')
adc = nibabel.load('mri/pt_0000/adc.nii')
ktrans = nibabel.load('mri/pt_0000/ktrans.nii')

print("T2W Image shape:", t2w.shape)
print("ADC Image shape:", adc.shape)
print("KTrans Image shape:", ktrans.shape)

t2w_img = t2w.get_fdata()
adc_img = adc.get_fdata()
ktrans_img = ktrans.get_fdata()

#print("\nT2W image-array", t2w_img)
#print("\nADC image-array", adc_img)
#print("\nKTrans image-array", ktrans_img)

plt.imshow(t2w_img[:,:,9], cm.gray)
plt.show()

plt.imshow(adc_img[:,:,9], cm.gray)
plt.show()

plt.imshow(ktrans_img[:,:,9], cm.gray)
plt.show()

mask = t2w_img.copy()
mask[:,:,:] = 0
mask[162:172, 219:229, 9] = 255
plt.imshow(mask[:,:,9], cm.gray)
plt.show()

plt.imshow(mask[:,:,9] * 0.9 + t2w_img[:,:,9] * 0.1)
plt.show()

vis = skimage.io.imread('vis/true/1.tiff')
vis = vis / 2**16

plt.imshow(vis)
plt.show()
