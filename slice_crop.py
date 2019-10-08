import csv
import nibabel as nib
import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.patches as patches # Testing
import skimage
import cv2 # For resizing

# Removing warning message
import imageio.core.util

def ignore_warnings(*args, **kwargs):
        pass

imageio.core.util._precision_warn = ignore_warnings

with open('info.csv') as csv_file:
    reader = csv.reader(csv_file)
    lines = list(reader)

lines.pop(0) # Removing the header

not_in_crop = 0

orig_height = int(input("crop height: "))
orig_width = int(input("crop width: "))

index = 0
while index < len(lines):
    entry = lines[index]
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
    
    # Specifying height and width
    crop_height = orig_height
    crop_width = orig_width

    if patient_id == "pt_0181":
        crop_height = int(input("\ncrop height for pt_0181: ")) # 191
        crop_width = int(input("crop width for pt_0181: ")) # 191
    elif patient_id == "pt_0075":
        crop_height = int(input("\ncrop height for pt_0075: ")) # 271
        crop_width = int(input("crop width for pt_0075: ")) # 271
    
    h = int((crop_height - 1) / 2)
    w = int((crop_width - 1) / 2)

    #t2w_image_cropped = t2w_image_sliced[i - 40 : i + 40, j - 40: j + 40]
    t2w_image_cropped = t2w_image_sliced[h_mid - h : h_mid + h, w_mid - w: w_mid + w]
    
    # Scaling to input height and width
    if patient_id == "pt_0181" or patient_id == "pt_0075":
        t2w_image_cropped = cv2.resize(t2w_image_cropped, dsize = (orig_width, orig_height), interpolation = cv2.INTER_NEAREST)

    if not (i > h_mid - h and i < h_mid + h and j > w_mid - w and j < w_mid + w)\
            or patient_id in ["pt_0181", "pt_0075"]:
        
        if patient_id not in ["pt_0181", "pt_0075"]:
            not_in_crop += 1
    
            print("ijk not in crop for:\npatient {},\nfid {}".format(patient_id, fid))
        
        mask = t2w_image_sliced.copy()
        mask[:,:] = 0
        mask[i - 2 : i + 2, j - 2 : j + 2] = 255
        fig1, (ax1, ax2) = plt.subplots(1, 2)
        #plt.subplot(1,2,1)
        ax1.imshow(mask*0.7 + t2w_image_sliced*0.3, cm.gray)
        rect = patches.Rectangle((h_mid - h, w_mid - w), crop_height, crop_width, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax1.add_patch(rect)
        #plt.subplot(1,2,2)
        ax2.imshow(t2w_image_cropped, cm.gray)
        plt.show()
        
    if patient_id in ["pt_0181", "pt_0075"] and input("good? ") == "no":
        index -= 1    

    # Saving cropped image
    save_path = 'mri/{}/t2w_cropped_0{}.nii'.format(patient_id, fid)
    skimage.io.imsave(save_path, t2w_image_cropped)
    
    # Saving cropped image as png
    save_path = 'test/crop_t2w/{}_{}.png'.format(patient_id, fid)
    skimage.io.imsave(save_path, t2w_image_cropped)

    index += 1
    
    print("{}/326 done".format(index))

print(not_in_crop, "ijk's not making the crop")
