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

with open('../data/info.csv') as csv_file:
    reader = csv.reader(csv_file)
    lines = list(reader)

lines.pop(0) # Removing the header

def extract_image(entry, path_name, crop_height, crop_width):    
    ijk_in_crop = True

    # Extracting important fields
    patient_id = entry[1]
    i, j, k = [int(x) for x in entry[4].split(' ')]
    fid = int(entry[3])
    t2w_height, t2w_width = [int(x) for x in entry[-2].split('x')[:2]]

    # Loading each image and locating the required slice
    img_file = nib.load('../data/mri/{}/{}.nii'.format(patient_id, path_name))
    image = img_file.get_fdata()
    image_sliced = image[:,:,k]

    # Saving the sliced image
    save_path = '../data/mri/{}/{}_sliced_0{}.nii'.format(patient_id, path_name, fid)
    skimage.io.imsave(save_path, image_sliced)
    
    # Cropping the image in the centre
    image_h, image_w = image_sliced.shape 
    
    # Scaling the (i,j) accordingly
    if t2w_height != 384 or t2w_width != 384:
        i = int(i * (384 / t2w_height))
        j = int(j * (384 / t2w_width))

  #  print(i,j, image_h, image_w)
    # If the image height or width doesn't equal 384 scale it to that size
    if image_h != 384 or image_w != 384:
        image_sliced = cv2.resize(image_sliced, dsize = (384, 384), interpolation = cv2.INTER_NEAREST)

    h_mid = len(image_sliced) // 2
    w_mid = len(image_sliced[0]) // 2
    
    #orig_height = crop_height
    #orig_width = crop_width

    #if patient_id == "pt_0181":
        #crop_height = int(input("\ncrop height for pt_0181: ")) # 191    
        #crop_width = int(input("crop width for pt_0181: ")) # 191
     #   crop_height = 191
      #  crop_width = 191
    #elif patient_id == "pt_0075":
        #crop_height = int(input("\ncrop height for pt_0075: ")) # 271
        #crop_width = int(input("crop width for pt_0075: ")) # 271
     #   crop_height = 271
      #  crop_width = 271

    h = int((crop_height - 1) / 2)
    w = int((crop_width - 1) / 2)

    image_cropped = image_sliced[h_mid - h : h_mid + h + 1, w_mid - w: w_mid + w + 1]
    
    # Scaling to input height and width
    #if patient_id == "pt_0181" or patient_id == "pt_0075":
     #   image_cropped = cv2.resize(image_cropped, dsize = (orig_width, orig_height), interpolation = cv2.INTER_NEAREST)

    if not (i > h_mid - h and i < h_mid + h and j > w_mid - w and j < w_mid + w):
#            or patient_id in ["pt_0181", "pt_0075"]:
        print(i, j)
        print(h_mid, h)
        print(w_mid, w)
    #    if patient_id not in ["pt_0181", "pt_0075"]:
        ijk_in_crop = False

        print("ijk not in crop for:\npatient {},\nfid {}".format(patient_id, fid))
        
        mask = image_sliced.copy()
        mask[:,:] = 0
        mask[i - 2 : i + 2, j - 2 : j + 2] = 255
        fig1, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(mask * 0.7 + image_sliced * 0.3, cm.gray)
        rect = patches.Rectangle((h_mid - h, w_mid - w), crop_height, crop_width, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax1.add_patch(rect)
        ax2.imshow(image_cropped, cm.gray)
        plt.show()
        
#    if patient_id in ["pt_0181", "pt_0075"] and input("good? ") == "no":
 #       return None

    # Saving cropped image
    save_path = '../data/mri/{}/{}_cropped_0{}.nii'.format(patient_id, path_name, fid)
    skimage.io.imsave(save_path, image_cropped)
    
    # Saving cropped image as png
    save_path = '../data/test/crop_{}/{}_{}.png'.format(path_name, patient_id, fid)
    skimage.io.imsave(save_path, image_cropped)

    
    return ijk_in_crop


#orig_height = int(input("crop height: ")) # 143
#orig_width = int(input("crop width: ")) # 143
orig_height = 143
orig_width = 143

not_in_crop = 0

index = 0
while index < len(lines):
    entry = lines[index]
    for img_name in ["t2w", "adc", "ktrans"]:
        ijk_in_crop = extract_image(entry, img_name, orig_height, orig_width)

        if not ijk_in_crop:
            not_in_crop += 1     
    
  #  if ijk_in_crop != None:
    index += 1

    print("{}/326 done".format(index))

print(not_in_crop, "ijk's not making the crop")
