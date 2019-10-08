import csv
import nibabel as nib
import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.patches as patches # Testing
import skimage

with open('info.csv') as csv_file:
    reader = csv.reader(csv_file)
    lines = list(reader)

lines.pop(0) # Removing the header

for entry in lines:
    # Extracting important fields
    patient_id = entry[1]
    i, j, k = [int(x) for x in entry[4].split(' ')]
    fid = int(entry[3])

    # Loading each t2w image and locating the required slice
    t2w_file = nib.load('mri/{}/t2w.nii'.format(patient_id))
    t2w_image = t2w_file.get_fdata()
    t2w_image_sliced = t2w_image[:,:,k]

    # Saving the sliced image
    save_path = 'mri/{}/t2w_sliced_0{}.nii'.format(patient_id, fid)
    skimage.io.imsave(save_path, t2w_image_sliced)
    
    # Cropping the image in the centre
    h_mid = len(t2w_image_sliced) // 2
    w_mid = len(t2w_image_sliced[0]) // 2
    #t2w_image_cropped = t2w_image_sliced[i - 40 : i + 40, j - 40: j + 40]
    t2w_image_cropped = t2w_image_sliced[h_mid - 120 : h_mid + 120, w_mid - 120: w_mid + 120]

    if not (i > h_mid - 120 and i < h_mid + 120 and j > w_mid - 120 and j < w_mid + 120):
        print("ijk not in crop")

        mask = t2w_image_sliced.copy()
        mask[:,:] = 0
        mask[i - 2 : i + 2, j - 2 : j + 2] = 255
        fig1, (ax1, ax2) = plt.subplots(1, 2)
        #plt.subplot(1,2,1)
        ax1.imshow(mask*0.7 + t2w_image_sliced*0.3, cm.gray)
        rect = patches.Rectangle((h_mid - 120, w_mid - 120), 241, 241, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax1.add_patch(rect)
        #plt.subplot(1,2,2)
        ax2.imshow(t2w_image_cropped, cm.gray)
        plt.show()

    # Saving cropped image
    save_path = 'mri/{}/t2w_cropped_0{}.nii'.format(patient_id, fid)
    skimage.io.imsave(save_path, t2w_image_cropped)
    save_path = 'test/crop_t2w/{}_{}.png'.format(patient_id, fid)
    skimage.io.imsave(save_path, t2w_image_cropped)
