import tensorflow as tf
import tensorflow_datasets as tfds
import patientkeyinfo as pki
import SimpleITK as sitk
import pix2pix3D as pix2pix
import numpy as np
import scipy
from scipy import stats
from scipy import ndimage
from skimage.segmentation import flood, flood_fill
import psutil
import gc
import elastic_augmentation as ea
import sys
import os
import glob
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output



# Initialize variables.
EPOCHS = 150
scale_factor = 1
target_scale = 192
size_offset = 0

np.random.seed(0)
tf.random.set_seed(0)

LAMBDA = 10 
LAMBDA_B = 0.05 

learning_rate_disc_g = 2e-4 
learning_rate_gen_g = 2e-4 

learning_rate_disc_f = 2e-4 
learning_rate_gen_f = 2e-4 

daslice = 32 # Image slice to visualize in outputs.

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Cross validation fold - no cross validation for this work (train-val-test split instead).

if(len(sys.argv)>1):
    fold = int(sys.argv[1])
else:
    fold = 0

print("The fold " + str(fold))

folder_prepend = "./evals/20_ours/fold_"+str(fold)+"/" # Output folder, must exist.

# File name variables

images_mr_names = []
images_ct_names = []
images_mr_names_flipped = []
images_ct_names_flipped = []
val_images_mr_names = []
val_images_ct_names = []
test_images_mr_names = []
test_images_ct_names = []


# --- Define functions ---

def load_vol_ITK(datafile):
    """
    load volume file
    formats: everything SimpleITK is able to read
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(datafile)
    itkImage = reader.Execute()

    return itkImage


def checkIfDuplicates(listOfElems):
    ''' Check if given list contains any duplicates '''    
    setOfElems = set()
    for elem in listOfElems:
        if elem in setOfElems:
            return True
        else:
            setOfElems.add(elem)         
    return False

# The path with the data 
scratch_path = "../../../../../scratch/alexgumu/"


# size limit, same number of datasets on each
size_a = len(glob.glob('./data_to_interpolate/stroke_project_data/images_mr/*.nii.gz'))
size_b = len(glob.glob('./data_to_interpolate/stroke_project_data/images_ct/*.nii.gz'))
print(str(size_a) + " - " + str(size_b))

if size_a<size_b:
    size_lim = size_a
else:
    size_lim = size_b


os.makedirs(scratch_path+'np_datasets/fold_'+str(fold), exist_ok=True) # Create folder to store pre-processed 

# If the folder has data, skip pre-processing, otherwise pre-process.
if not os.path.exists(scratch_path+'np_datasets/fold_'+str(fold)+'/images_mr_anup.npy'):

    # source folder, csv keys, seed, num of train samples
    patients_mr = pki.PatientKeyInfo('./data_to_interpolate/stroke_project_data/images_mr/*.nii.gz', 'none', 10, -1, size_limit=size_lim, fold=fold)
    print(patients_mr.train)
    print(patients_mr.val)
    print(patients_mr.test)

    train_vol_names_mr = patients_mr.train
    val_vol_names_mr = patients_mr.val
    test_vol_names_mr = patients_mr.test

    result = checkIfDuplicates(train_vol_names_mr+val_vol_names_mr+test_vol_names_mr)
    if result:
        print('Yes, MR images list contains duplicates')
    else:
        print('No duplicates found in MR images list') 

    # source folder, csv keys, seed, num of train samples
    patients_ct = pki.PatientKeyInfo('./data_to_interpolate/stroke_project_data/images_ct/*.nii.gz', 'none', 10, -1, size_limit=size_lim, fold=fold)
    print(patients_ct.train)
    print(patients_ct.val)
    print(patients_ct.test)

    train_vol_names_ct = patients_ct.train
    val_vol_names_ct = patients_ct.val
    test_vol_names_ct = patients_ct.test

    result = checkIfDuplicates(train_vol_names_ct+val_vol_names_ct+test_vol_names_ct)
    if result:
        print('Yes, CT images list contains duplicates')
    else:
        print('No duplicates found in CT images list') 

    # Ground truth MR - source folder, csv keys, seed, num of train samples
    patients_mr_lesion = pki.PatientKeyInfo('./data_to_interpolate/stroke_project_data/grounds_mr/*.nii.gz', 'none', 10, -1, size_limit=size_lim, fold=fold)
    print(patients_mr_lesion.train)
    print(patients_mr_lesion.val)
    print(patients_mr_lesion.test)

    ground_train_vol_names_mr = patients_mr_lesion.train
    ground_val_vol_names_mr = patients_mr_lesion.val
    ground_test_vol_names_mr = patients_mr_lesion.test

    result = checkIfDuplicates(ground_train_vol_names_mr+ground_val_vol_names_mr+ground_test_vol_names_mr)
    if result:
        print('Yes, MR grounds list contains duplicates')
    else:
        print('No duplicates found in MR grounds list') 

    # Ground truth CT - source folder, csv keys, seed, num of train samples
    patients_ct_lesion = pki.PatientKeyInfo('./data_to_interpolate/stroke_project_data/grounds_ct/*.nii.gz', 'none', 10, -1, size_limit=size_lim, fold=fold)
    print(patients_ct_lesion.train)
    print(patients_ct_lesion.val)
    print(patients_ct_lesion.test)

    ground_train_vol_names_ct = patients_ct_lesion.train
    ground_val_vol_names_ct = patients_ct_lesion.val
    ground_test_vol_names_ct = patients_ct_lesion.test

    result = checkIfDuplicates(ground_train_vol_names_ct+ground_val_vol_names_ct+ground_test_vol_names_ct)
    if result:
        print('Yes, CT grounds list contains duplicates')
    else:
        print('No duplicates found in CT grounds list') 



    # MR training data
    first_flag = True
    for idx in range(len(train_vol_names_mr)):

        # Load volumes and ground truth lesion masks. Place in cropped frame 64 x 160 x 160. 
        print("Real images idx: "+str(idx))
        itkImage_mr = load_vol_ITK(train_vol_names_mr[idx])
        Xmr_vol = sitk.GetArrayFromImage(itkImage_mr).astype('float32')
        num_slices = int(Xmr_vol.shape[0])

        gnd_itkImage_mr = load_vol_ITK(ground_train_vol_names_mr[idx])
        gnd_Xmr_vol = sitk.GetArrayFromImage(gnd_itkImage_mr).astype('float32')

        Xmr = Xmr_vol     

        if scale_factor != 1:
            Xmr = scipy.ndimage.interpolation.zoom(Xmr, scale_factor, order=1)

        Xmr_resized = np.zeros(np.array([64, 160, 160]) + size_offset, np.float32)
        Xmr_resized = Xmr[80:144, 16:176, 16:176]

        # standarize and normalize
        damean = stats.tmean(Xmr_vol[Xmr_vol>10])
        dastd = stats.tstd(Xmr_vol[Xmr_vol>10])
        damin = (damean-(3*dastd))
        if damin <= np.min(Xmr_vol):
            damin = np.min(Xmr_vol)
        damax = (damean+(3*dastd))
        if damax >= np.max(Xmr_vol):
            damax = np.max(Xmr_vol)

        Xmr_resized = np.where(Xmr_resized <= damin, damin, Xmr_resized)
        Xmr_resized = np.where(Xmr_resized >= damax, damax, Xmr_resized)
        Xmr_resized = np.interp(Xmr_resized, (damin, damax), (-1, 1))
        Xmr_flipped = np.copy(Xmr_resized)
        
        # ground truth
        gnd_Xmr = gnd_Xmr_vol
        
        if scale_factor != 1:
            gnd_Xmr = scipy.ndimage.interpolation.zoom(gnd_Xmr, scale_factor, order=0)

        gnd_Xmr_resized = np.zeros(np.array([64, 160, 160]) + size_offset, np.float32)
        gnd_Xmr_resized = gnd_Xmr[80:144, 16:176, 16:176]
        
        gnd_Xmr_resized = np.where(gnd_Xmr_resized>0.0, 1.0, 0.0)
        gnd_Xmr_flipped = np.copy(gnd_Xmr_resized) 

        binary_Xmr = np.where(Xmr_resized>-0.9, 1.0, 0.0) 

        # mask background
        filled_Xmr_resized = flood_fill(binary_Xmr, (5, 5, 5), 100.0)   
        Xmr_background = np.where(filled_Xmr_resized>=100.0, 1.0, 0.0)
        Xmr_background_flipped = np.copy(Xmr_background)           

        if np.sum(gnd_Xmr) >= 10:
            if first_flag:

                Xmr_resized = Xmr_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                images_mr = Xmr_resized.astype('float32')
                Xmr_flipped_3d = np.flip(Xmr_flipped, 2)
                Xmr_flipped_3d = Xmr_flipped_3d[np.newaxis, np.newaxis, ..., np.newaxis]
                images_mr_flipped = Xmr_flipped_3d.astype('float32')

                # ground truths
                gnd_Xmr_resized = gnd_Xmr_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                gnd_images_mr = gnd_Xmr_resized.astype('float32')
                gnd_Xmr_flipped_3d = np.flip(gnd_Xmr_flipped, 2)
                gnd_Xmr_flipped_3d = gnd_Xmr_flipped_3d[np.newaxis, np.newaxis, ..., np.newaxis]
                gnd_images_mr_flipped = gnd_Xmr_flipped_3d.astype('float32')

                # background masks
                Xmr_background = Xmr_background[np.newaxis, np.newaxis, ..., np.newaxis]
                back_images_mr = Xmr_background.astype('float32')
                back_Xmr_flipped_3d = np.flip(Xmr_background_flipped, 2)
                back_Xmr_flipped_3d = back_Xmr_flipped_3d[np.newaxis, np.newaxis, ..., np.newaxis]
                back_images_mr_flipped = back_Xmr_flipped_3d.astype('float32')
                
                images_mr_names.append(train_vol_names_mr[idx])
                images_mr_names_flipped.append(train_vol_names_mr[idx]+"_flipped")

                first_flag = False

            else:

                Xmr_resized = Xmr_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                images_mr = np.append(images_mr, Xmr_resized.astype('float32'), axis=0)
                Xmr_flipped_3d = np.flip(Xmr_flipped, 2)
                Xmr_flipped_3d = Xmr_flipped_3d[np.newaxis, np.newaxis, ..., np.newaxis]
                images_mr_flipped = np.append(images_mr_flipped, Xmr_flipped_3d.astype('float32'), axis=0)

                # ground truths
                gnd_Xmr_resized = gnd_Xmr_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                gnd_images_mr = np.append(gnd_images_mr, gnd_Xmr_resized.astype('float32'), axis=0)
                gnd_Xmr_flipped_3d = np.flip(gnd_Xmr_flipped, 2)
                gnd_Xmr_flipped_3d = gnd_Xmr_flipped_3d[np.newaxis, np.newaxis, ..., np.newaxis]
                gnd_images_mr_flipped = np.append(gnd_images_mr_flipped, gnd_Xmr_flipped_3d.astype('float32'), axis=0)

                # background masks
                Xmr_background = Xmr_background[np.newaxis, np.newaxis, ..., np.newaxis]
                back_images_mr = np.append(back_images_mr, Xmr_background.astype('float32'), axis=0)
                back_Xmr_flipped_3d = np.flip(Xmr_background_flipped, 2)
                back_Xmr_flipped_3d = back_Xmr_flipped_3d[np.newaxis, np.newaxis, ..., np.newaxis]
                back_images_mr_flipped = np.append(back_images_mr_flipped, back_Xmr_flipped_3d.astype('float32'), axis=0)
                
                images_mr_names.append(train_vol_names_mr[idx])
                images_mr_names_flipped.append(train_vol_names_mr[idx]+"_flipped")



    # CT training data
    first_flag = True
    for idx in range(len(train_vol_names_ct)):

        # Load volumes and ground truth lesion masks. Place in cropped frame 64 x 160 x 160. 
        print("Real images idx: "+str(idx))
        itkImage_ct = load_vol_ITK(train_vol_names_ct[idx])
        Xct_vol = sitk.GetArrayFromImage(itkImage_ct).astype('float32')
        num_slices = int(Xct_vol.shape[0])

        gnd_itkImage_ct = load_vol_ITK(ground_train_vol_names_ct[idx])
        gnd_Xct_vol = sitk.GetArrayFromImage(gnd_itkImage_ct).astype('float32')

        Xct = Xct_vol

        if scale_factor != 1:
            Xct = scipy.ndimage.interpolation.zoom(Xct, scale_factor, order=1)

        Xct_resized = np.zeros(np.array([64, 160, 160]) + size_offset, np.float32)
        Xct_resized = Xct[80:144, 16:176, 16:176]
        
        # standarize and normalize
        damean = stats.tmean(Xct_vol[Xct_vol>0.1])
        dastd = stats.tstd(Xct_vol[Xct_vol>0.1])
        damin = (damean-(3*dastd))
        if damin <= np.min(Xct_vol):
            damin = np.min(Xct_vol)
        damax = (damean+(3*dastd))
        if damax >= np.max(Xct_vol):
            damax = np.max(Xct_vol)

        Xct_resized = np.where(Xct_resized <= damin, damin, Xct_resized)
        Xct_resized = np.where(Xct_resized >= damax, damax, Xct_resized)
        Xct_resized = np.interp(Xct_resized, (damin, damax), (-1, 1))

        Xct_flipped = np.copy(Xct_resized)

        # ground truth
        gnd_Xct = gnd_Xct_vol
        
        if scale_factor != 1:
            gnd_Xct = scipy.ndimage.interpolation.zoom(gnd_Xct, scale_factor, order=0)

        gnd_Xct_resized = np.zeros(np.array([64, 160, 160]) + size_offset, np.float32)
        gnd_Xct_resized = gnd_Xct[80:144, 16:176, 16:176]

        gnd_Xct_resized = np.where(gnd_Xct_resized>0.0, 1.0, 0.0)
        gnd_Xct_flipped = np.copy(gnd_Xct_resized)            

        binary_Xct = np.where(Xct_resized>-0.9, 1.0, 0.0) 

        # mask background
        filled_Xct_resized = flood_fill(binary_Xct, (5, 5, 5), 100.0)   
        Xct_background = np.where(filled_Xct_resized>=100.0, 1.0, 0.0)
        Xct_background_flipped = np.copy(Xct_background)        

        if np.sum(gnd_Xct) >= 10:

            if first_flag:

                Xct_resized = Xct_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                images_ct = Xct_resized.astype('float32')
                Xct_flipped_3d = np.flip(Xct_flipped, 2)
                Xct_flipped_3d = Xct_flipped_3d[np.newaxis, np.newaxis, ..., np.newaxis]
                images_ct_flipped = Xct_flipped_3d.astype('float32')

                # ground truths
                gnd_Xct_resized = gnd_Xct_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                gnd_images_ct = gnd_Xct_resized.astype('float32')
                gnd_Xct_flipped_3d = np.flip(gnd_Xct_flipped, 2)
                gnd_Xct_flipped_3d = gnd_Xct_flipped_3d[np.newaxis, np.newaxis, ..., np.newaxis]
                gnd_images_ct_flipped = gnd_Xct_flipped_3d.astype('float32')

                # background masks
                Xct_background = Xct_background[np.newaxis, np.newaxis, ..., np.newaxis]
                back_images_ct = Xct_background.astype('float32')
                back_Xct_flipped_3d = np.flip(Xct_background_flipped, 2)
                back_Xct_flipped_3d  = back_Xct_flipped_3d[np.newaxis, np.newaxis, ..., np.newaxis]
                back_images_ct_flipped = back_Xct_flipped_3d.astype('float32')
                
                images_ct_names.append(train_vol_names_ct[idx])
                images_ct_names_flipped.append(train_vol_names_ct[idx]+"_flipped")

                first_flag = False

            else:

                Xct_resized = Xct_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                images_ct = np.append(images_ct, Xct_resized.astype('float32'), axis=0)
                Xct_flipped_3d = np.flip(Xct_flipped, 2)
                Xct_flipped_3d = Xct_flipped_3d[np.newaxis, np.newaxis, ..., np.newaxis]
                images_ct_flipped = np.append(images_ct_flipped, Xct_flipped_3d.astype('float32'), axis=0)

                # ground truths
                gnd_Xct_resized = gnd_Xct_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                gnd_images_ct = np.append(gnd_images_ct, gnd_Xct_resized.astype('float32'), axis=0)
                gnd_Xct_flipped_3d = np.flip(gnd_Xct_flipped, 2)
                gnd_Xct_flipped_3d = gnd_Xct_flipped_3d[np.newaxis, np.newaxis, ..., np.newaxis]
                gnd_images_ct_flipped = np.append(gnd_images_ct_flipped, gnd_Xct_flipped_3d.astype('float32'), axis=0)

                # background masks
                Xct_background = Xct_background[np.newaxis, np.newaxis, ..., np.newaxis]
                back_images_ct = np.append(back_images_ct, Xct_background.astype('float32'), axis=0)
                back_Xct_flipped_3d = np.flip(Xct_background_flipped, 2)
                back_Xct_flipped_3d  = back_Xct_flipped_3d[np.newaxis, np.newaxis, ..., np.newaxis]
                back_images_ct_flipped = np.append(back_images_ct_flipped, back_Xct_flipped_3d.astype('float32'), axis=0)
                
                images_ct_names.append(train_vol_names_ct[idx])
                images_ct_names_flipped.append(train_vol_names_ct[idx]+"_flipped")


    # MR validation data
    first_flag = True
    for idx in range(len(val_vol_names_mr)):

        print("Real images idx: "+str(idx))
        itkImage_mr = load_vol_ITK(val_vol_names_mr[idx])
        Xmr_vol = sitk.GetArrayFromImage(itkImage_mr).astype('float32')
        num_slices = int(Xmr_vol.shape[0])

        gnd_itkImage_mr = load_vol_ITK(ground_val_vol_names_mr[idx])
        gnd_Xmr_vol = sitk.GetArrayFromImage(gnd_itkImage_mr).astype('float32')

        Xmr = Xmr_vol     

        if scale_factor != 1:
            Xmr = scipy.ndimage.interpolation.zoom(Xmr, scale_factor, order=1)

        Xmr_resized = np.zeros(np.array([64, 160, 160]) + size_offset, np.float32)
        Xmr_resized = Xmr[80:144, 16:176, 16:176]

        # standarize and normalize
        damean = stats.tmean(Xmr_vol[Xmr_vol>10])
        dastd = stats.tstd(Xmr_vol[Xmr_vol>10])
        damin = (damean-(3*dastd))
        if damin <= np.min(Xmr_vol):
            damin = np.min(Xmr_vol)
        damax = (damean+(3*dastd))
        if damax >= np.max(Xmr_vol):
            damax = np.max(Xmr_vol)

        Xmr_resized = np.where(Xmr_resized <= damin, damin, Xmr_resized)
        Xmr_resized = np.where(Xmr_resized >= damax, damax, Xmr_resized)
        Xmr_resized = np.interp(Xmr_resized, (damin, damax), (-1, 1))

        #Xmr_resized = hist_match(Xmr_resized, Xmr_atlas_resized)
        
        # ground truth
        gnd_Xmr = gnd_Xmr_vol
        
        if scale_factor != 1:
            gnd_Xmr = scipy.ndimage.interpolation.zoom(gnd_Xmr, scale_factor, order=0)

        gnd_Xmr_resized = np.zeros(np.array([64, 160, 160]) + size_offset, np.float32)
        gnd_Xmr_resized = gnd_Xmr[80:144, 16:176, 16:176]
        
        gnd_Xmr_resized = np.where(gnd_Xmr_resized>0.0, 1.0, 0.0)
        binary_Xmr = np.where(Xmr_resized>-0.9, 1.0, 0.0) 

        # mask background
        filled_Xmr_resized = flood_fill(binary_Xmr, (5, 5, 5), 100.0)   
        Xmr_background = np.where(filled_Xmr_resized>=100.0, 1.0, 0.0)   

        if np.sum(gnd_Xmr) >= 10:

            if first_flag:

                Xmr_resized = Xmr_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                val_images_mr = Xmr_resized.astype('float32')

                # ground truths
                gnd_Xmr_resized = gnd_Xmr_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                val_gnd_images_mr = gnd_Xmr_resized.astype('float32')

                # background masks
                Xmr_background = Xmr_background[np.newaxis, np.newaxis, ..., np.newaxis]
                val_back_images_mr = Xmr_background.astype('float32')
                
                val_images_mr_names.append(val_vol_names_mr[idx])

                first_flag = False

            else:

                Xmr_resized = Xmr_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                val_images_mr = np.append(val_images_mr, Xmr_resized.astype('float32'), axis=0)

                # ground truths
                gnd_Xmr_resized = gnd_Xmr_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                val_gnd_images_mr = np.append(val_gnd_images_mr, gnd_Xmr_resized.astype('float32'), axis=0)

                # background masks
                Xmr_background = Xmr_background[np.newaxis, np.newaxis, ..., np.newaxis]
                val_back_images_mr = np.append(val_back_images_mr, Xmr_background.astype('float32'), axis=0)
                
                val_images_mr_names.append(val_vol_names_mr[idx])




    # CT validation data
    first_flag = True
    for idx in range(len(val_vol_names_ct)):

        print("Real images idx: "+str(idx))
        itkImage_ct = load_vol_ITK(val_vol_names_ct[idx])
        Xct_vol = sitk.GetArrayFromImage(itkImage_ct).astype('float32')
        num_slices = int(Xct_vol.shape[0])

        gnd_itkImage_ct = load_vol_ITK(ground_val_vol_names_ct[idx])
        gnd_Xct_vol = sitk.GetArrayFromImage(gnd_itkImage_ct).astype('float32')

        Xct = Xct_vol

        if scale_factor != 1:
            Xct = scipy.ndimage.interpolation.zoom(Xct, scale_factor, order=1)

        Xct_resized = np.zeros(np.array([64, 160, 160]) + size_offset, np.float32)
        Xct_resized = Xct[80:144, 16:176, 16:176]
        
        # standarize and normalize
        damean = stats.tmean(Xct_vol[Xct_vol>0.1])
        dastd = stats.tstd(Xct_vol[Xct_vol>0.1])
        damin = (damean-(3*dastd))
        if damin <= np.min(Xct_vol):
            damin = np.min(Xct_vol)
        damax = (damean+(3*dastd))
        if damax >= np.max(Xct_vol):
            damax = np.max(Xct_vol)

        Xct_resized = np.where(Xct_resized <= damin, damin, Xct_resized)
        Xct_resized = np.where(Xct_resized >= damax, damax, Xct_resized)
        Xct_resized = np.interp(Xct_resized, (damin, damax), (-1, 1))

        # ground truth
        gnd_Xct = gnd_Xct_vol
        
        if scale_factor != 1:
            gnd_Xct = scipy.ndimage.interpolation.zoom(gnd_Xct, scale_factor, order=0)

        gnd_Xct_resized = np.zeros(np.array([64, 160, 160]) + size_offset, np.float32)
        gnd_Xct_resized = gnd_Xct[80:144, 16:176, 16:176]

        gnd_Xct_resized = np.where(gnd_Xct_resized>0.0, 1.0, 0.0)
        binary_Xct = np.where(Xct_resized>-0.9, 1.0, 0.0) 

        # mask background
        filled_Xct_resized = flood_fill(binary_Xct, (5, 5, 5), 100.0)   
        Xct_background = np.where(filled_Xct_resized>=100.0, 1.0, 0.0)
        Xct_background_flipped = np.copy(Xct_background)             

        if np.sum(gnd_Xct) >= 10:

            if first_flag:

                Xct_resized = Xct_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                val_images_ct = Xct_resized.astype('float32')

                # ground truths
                gnd_Xct_resized = gnd_Xct_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                val_gnd_images_ct = gnd_Xct_resized.astype('float32')

                # background masks
                Xct_background = Xct_background[np.newaxis, np.newaxis, ..., np.newaxis]
                val_back_images_ct = Xct_background.astype('float32')
                
                val_images_ct_names.append(val_vol_names_ct[idx])

                first_flag = False

            else:

                Xct_resized = Xct_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                val_images_ct = np.append(val_images_ct, Xct_resized.astype('float32'), axis=0)

                # ground truths
                gnd_Xct_resized = gnd_Xct_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                val_gnd_images_ct = np.append(val_gnd_images_ct, gnd_Xct_resized.astype('float32'), axis=0)

                # background masks
                Xct_background = Xct_background[np.newaxis, np.newaxis, ..., np.newaxis]
                val_back_images_ct = np.append(val_back_images_ct, Xct_background.astype('float32'), axis=0)
                
                val_images_ct_names.append(val_vol_names_ct[idx])


    # MR test data
    first_flag = True
    for idx in range(len(test_vol_names_mr)):

        print("Real images idx: "+str(idx))
        itkImage_mr = load_vol_ITK(test_vol_names_mr[idx])
        Xmr_vol = sitk.GetArrayFromImage(itkImage_mr).astype('float32')
        num_slices = int(Xmr_vol.shape[0])

        gnd_itkImage_mr = load_vol_ITK(ground_test_vol_names_mr[idx])
        gnd_Xmr_vol = sitk.GetArrayFromImage(gnd_itkImage_mr).astype('float32')

        Xmr = Xmr_vol     

        if scale_factor != 1:
            Xmr = scipy.ndimage.interpolation.zoom(Xmr, scale_factor, order=1)

        Xmr_resized = np.zeros(np.array([64, 160, 160]) + size_offset, np.float32)
        Xmr_resized = Xmr[80:144, 16:176, 16:176]

        # standarize and normalize
        damean = stats.tmean(Xmr_vol[Xmr_vol>10])
        dastd = stats.tstd(Xmr_vol[Xmr_vol>10])
        damin = (damean-(3*dastd))
        if damin <= np.min(Xmr_vol):
            damin = np.min(Xmr_vol)
        damax = (damean+(3*dastd))
        if damax >= np.max(Xmr_vol):
            damax = np.max(Xmr_vol)

        Xmr_resized = np.where(Xmr_resized <= damin, damin, Xmr_resized)
        Xmr_resized = np.where(Xmr_resized >= damax, damax, Xmr_resized)
        Xmr_resized = np.interp(Xmr_resized, (damin, damax), (-1, 1))

        #Xmr_resized = hist_match(Xmr_resized, Xmr_atlas_resized)
        
        # ground truth
        gnd_Xmr = gnd_Xmr_vol
        
        if scale_factor != 1:
            gnd_Xmr = scipy.ndimage.interpolation.zoom(gnd_Xmr, scale_factor, order=0)

        gnd_Xmr_resized = np.zeros(np.array([64, 160, 160]) + size_offset, np.float32)
        gnd_Xmr_resized = gnd_Xmr[80:144, 16:176, 16:176]
        
        gnd_Xmr_resized = np.where(gnd_Xmr_resized>0.0, 1.0, 0.0)
        binary_Xmr = np.where(Xmr_resized>-0.9, 1.0, 0.0) 

        # mask background
        filled_Xmr_resized = flood_fill(binary_Xmr, (5, 5, 5), 100.0)   
        Xmr_background = np.where(filled_Xmr_resized>=100.0, 1.0, 0.0)           

        if np.sum(gnd_Xmr) >= 10:

            if first_flag:

                Xmr_resized = Xmr_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                test_images_mr = Xmr_resized.astype('float32')

                # ground truths
                gnd_Xmr_resized = gnd_Xmr_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                test_gnd_images_mr = gnd_Xmr_resized.astype('float32')
                
                test_images_mr_names.append(test_vol_names_mr[idx])

                first_flag = False

            else:

                Xmr_resized = Xmr_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                test_images_mr = np.append(test_images_mr, Xmr_resized.astype('float32'), axis=0)

                # ground truths
                gnd_Xmr_resized = gnd_Xmr_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                test_gnd_images_mr = np.append(test_gnd_images_mr, gnd_Xmr_resized.astype('float32'), axis=0)
                
                test_images_mr_names.append(test_vol_names_mr[idx])




    # CT test data
    first_flag = True
    for idx in range(len(test_vol_names_ct)):

        print("Real images idx: "+str(idx))
        itkImage_ct = load_vol_ITK(test_vol_names_ct[idx])
        Xct_vol = sitk.GetArrayFromImage(itkImage_ct).astype('float32')
        num_slices = int(Xct_vol.shape[0])

        gnd_itkImage_ct = load_vol_ITK(ground_test_vol_names_ct[idx])
        gnd_Xct_vol = sitk.GetArrayFromImage(gnd_itkImage_ct).astype('float32')

        Xct = Xct_vol

        if scale_factor != 1:
            Xct = scipy.ndimage.interpolation.zoom(Xct, scale_factor, order=1)

        Xct_resized = np.zeros(np.array([64, 160, 160]) + size_offset, np.float32)
        Xct_resized = Xct[80:144, 16:176, 16:176]
        
        # standarize and normalize
        damean = stats.tmean(Xct_vol[Xct_vol>0.1])
        dastd = stats.tstd(Xct_vol[Xct_vol>0.1])
        damin = (damean-(3*dastd))
        if damin <= np.min(Xct_vol):
            damin = np.min(Xct_vol)
        damax = (damean+(3*dastd))
        if damax >= np.max(Xct_vol):
            damax = np.max(Xct_vol)

        Xct_resized = np.where(Xct_resized <= damin, damin, Xct_resized)
        Xct_resized = np.where(Xct_resized >= damax, damax, Xct_resized)
        Xct_resized = np.interp(Xct_resized, (damin, damax), (-1, 1))

        # ground truth
        gnd_Xct = gnd_Xct_vol
        
        if scale_factor != 1:
            gnd_Xct = scipy.ndimage.interpolation.zoom(gnd_Xct, scale_factor, order=0)

        gnd_Xct_resized = np.zeros(np.array([64, 160, 160]) + size_offset, np.float32)
        gnd_Xct_resized = gnd_Xct[80:144, 16:176, 16:176]

        gnd_Xct_resized = np.where(gnd_Xct_resized>0.0, 1.0, 0.0)
        binary_Xct = np.where(Xct_resized>-0.9, 1.0, 0.0) 

        # mask background
        filled_Xct_resized = flood_fill(binary_Xct, (5, 5, 5), 100.0)   
        Xct_background = np.where(filled_Xct_resized>=100.0, 1.0, 0.0)
        Xct_background_flipped = np.copy(Xct_background)             

        if np.sum(gnd_Xct) >= 10:

            if first_flag:

                Xct_resized = Xct_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                test_images_ct = Xct_resized.astype('float32')

                # ground truths
                gnd_Xct_resized = gnd_Xct_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                test_gnd_images_ct = gnd_Xct_resized.astype('float32')
                
                test_images_ct_names.append(test_vol_names_ct[idx])

                first_flag = False

            else:

                Xct_resized = Xct_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                test_images_ct = np.append(test_images_ct, Xct_resized.astype('float32'), axis=0)

                # ground truths
                gnd_Xct_resized = gnd_Xct_resized[np.newaxis, np.newaxis, ..., np.newaxis]
                test_gnd_images_ct = np.append(test_gnd_images_ct, gnd_Xct_resized.astype('float32'), axis=0)
                
                test_images_ct_names.append(test_vol_names_ct[idx])



    process_m = psutil.Process(os.getpid())
    print("A - RAM %.2f Gb" % (process_m.memory_info().rss/1000/1000/1000))


    # inspect images

    for z in range(len(images_mr)):
        base = (images_mr[z][0, daslice, :, :, 0]+1)*0.5
        overlay = np.array([base+(gnd_images_mr[z][0, daslice, :, :, 0]*0.15), base, base])
        overlay = np.moveaxis(overlay, 0, -1)
        plt.imshow(overlay)
        plt.title(images_mr_names[z])
        plt.savefig("inspect_images_anup/mr_img_"+str(z)+".png")
        plt.close()

    for z in range(len(images_ct)):
        base = (images_ct[z][0, daslice, :, :, 0]+1)*0.5
        overlay = np.array([base+(gnd_images_ct[z][0, daslice, :, :, 0]*0.15), base, base])
        overlay = np.moveaxis(overlay, 0, -1)
        plt.imshow(overlay)
        plt.title(images_ct_names[z])
        plt.savefig("inspect_images_anup/ct_img_"+str(z)+".png")
        plt.close()

    for z in range(len(val_images_mr)):
        base = (val_images_mr[z][0, daslice, :, :, 0]+1)*0.5
        overlay = np.array([base+(val_gnd_images_mr[z][0, daslice, :, :, 0]*0.15), base, base])
        overlay = np.moveaxis(overlay, 0, -1)
        plt.imshow(overlay)
        plt.title(val_images_mr_names[z])
        plt.savefig("inspect_images_anup/mr_val_img_"+str(z)+".png")
        plt.close()

    for z in range(len(val_images_ct)):
        base = (val_images_ct[z][0, daslice, :, :, 0]+1)*0.5
        overlay = np.array([base+(val_gnd_images_ct[z][0, daslice, :, :, 0]*0.15), base, base])
        overlay = np.moveaxis(overlay, 0, -1)
        plt.imshow(overlay)
        plt.title(val_images_ct_names[z])
        plt.savefig("inspect_images_anup/ct_val_img_"+str(z)+".png")
        plt.close()

    for z in range(len(test_images_mr)):
        base = (test_images_mr[z][0, daslice, :, :, 0]+1)*0.5
        overlay = np.array([base+(test_gnd_images_mr[z][0, daslice, :, :, 0]*0.15), base, base])
        overlay = np.moveaxis(overlay, 0, -1)
        plt.imshow(overlay)
        plt.title(test_images_mr_names[z])
        plt.savefig("inspect_images_anup/mr_test_img_"+str(z)+".png")
        plt.close()

    for z in range(len(test_images_ct)):
        base = (test_images_ct[z][0, daslice, :, :, 0]+1)*0.5
        overlay = np.array([base+(test_gnd_images_ct[z][0, daslice, :, :, 0]*0.15), base, base])
        overlay = np.moveaxis(overlay, 0, -1)
        plt.imshow(overlay)
        plt.title(test_images_ct_names[z])
        plt.savefig("inspect_images_anup/ct_test_img_"+str(z)+".png")
        plt.close()

    # Save to npy files

    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/images_mr_anup.npy', 'wb') as f:
        np.save(f, images_mr)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/images_ct_anup.npy', 'wb') as f:
        np.save(f, images_ct)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/gnd_images_mr_anup.npy', 'wb') as f:
        np.save(f, gnd_images_mr)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/gnd_images_ct_anup.npy', 'wb') as f:
        np.save(f, gnd_images_ct)

    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/images_mr_flipped_anup.npy', 'wb') as f:
        np.save(f, images_mr_flipped)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/images_ct_flipped_anup.npy', 'wb') as f:
        np.save(f, images_ct_flipped)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/gnd_images_mr_flipped_anup.npy', 'wb') as f:
        np.save(f, gnd_images_mr_flipped)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/gnd_images_ct_flipped_anup.npy', 'wb') as f:
        np.save(f, gnd_images_ct_flipped)

    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/val_images_mr_anup.npy', 'wb') as f:
        np.save(f, val_images_mr)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/val_images_ct_anup.npy', 'wb') as f:
        np.save(f, val_images_ct)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/val_gnd_images_mr_anup.npy', 'wb') as f:
        np.save(f, val_gnd_images_mr)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/val_gnd_images_ct_anup.npy', 'wb') as f:
        np.save(f, val_gnd_images_ct)

    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/test_images_mr_anup.npy', 'wb') as f:
        np.save(f, test_images_mr)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/test_images_ct_anup.npy', 'wb') as f:
        np.save(f, test_images_ct)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/test_gnd_images_mr_anup.npy', 'wb') as f:
        np.save(f, test_gnd_images_mr)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/test_gnd_images_ct_anup.npy', 'wb') as f:
        np.save(f, test_gnd_images_ct)

    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/back_images_mr_anup.npy', 'wb') as f:
        np.save(f, back_images_mr)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/val_back_images_mr_anup.npy', 'wb') as f:
        np.save(f, val_back_images_mr)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/back_images_ct_anup.npy', 'wb') as f:
        np.save(f, back_images_ct)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/val_back_images_ct_anup.npy', 'wb') as f:
        np.save(f, val_back_images_ct)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/back_images_mr_anup_flipped.npy', 'wb') as f:
        np.save(f, back_images_mr_flipped)
    with open(scratch_path+'np_datasets/fold_'+str(fold)+'/back_images_ct_anup_flipped.npy', 'wb') as f:
        np.save(f, back_images_ct_flipped)


else:

    # Load from npy files

    images_mr = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/images_mr_anup.npy')
    images_ct = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/images_ct_anup.npy')
    gnd_images_mr = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/gnd_images_mr_anup.npy')
    gnd_images_ct = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/gnd_images_ct_anup.npy')

    images_mr_flipped = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/images_mr_flipped_anup.npy')
    images_ct_flipped = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/images_ct_flipped_anup.npy')
    gnd_images_mr_flipped = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/gnd_images_mr_flipped_anup.npy')
    gnd_images_ct_flipped = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/gnd_images_ct_flipped_anup.npy')

    val_images_mr = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/val_images_mr_anup.npy')
    val_images_ct = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/val_images_ct_anup.npy')
    val_gnd_images_mr = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/val_gnd_images_mr_anup.npy')
    val_gnd_images_ct = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/val_gnd_images_ct_anup.npy')



print(images_ct.shape)
print(val_images_ct.shape)
print(images_mr.shape)
print(val_images_mr.shape)


process_m = psutil.Process(os.getpid())
print("post-dataload - RAM %.2f Gb" % (process_m.memory_info().rss/1000/1000/1000))

BATCH_SIZE = 1
IMG_WIDTH = 192
IMG_HEIGHT = 192
IMG_DEPTH = 192

sample_mr = images_mr[30]
sample_ct = images_ct[33]
sample_mr_gnd = gnd_images_mr[30]
sample_ct_gnd = gnd_images_ct[33]

sample_mr_test = val_images_mr[12]
sample_ct_test = val_images_ct[12]
sample_mr_gnd_test = val_gnd_images_mr[12]
sample_ct_gnd_test = val_gnd_images_ct[12]

print(sample_mr.shape)


OUTPUT_CHANNELS = 1

# Check if pre-trained model exists, otherwise load new ones.  
if not os.path.exists(folder_prepend+"generator_mr_to_ct.h5"):

    generator_g, mid_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
    generator_f, mid_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
    print(generator_g.summary())
    print(mid_g.summary())
     
    discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
    print(discriminator_x.summary())
    discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

else:

    generator_g, mid_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
    generator_f, mid_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
    
    generator_g.load_weights(folder_prepend+"generator_mr_to_ct.h5")
    generator_f.load_weights(folder_prepend+"generator_ct_to_mr.h5")

    discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
    discriminator_x.load_weights(folder_prepend+"discriminator_mr_to_ct.h5")

    discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
    discriminator_y.load_weights(folder_prepend+"discriminator_ct_to_mr.h5")


# Plot and save initial state of the models.
to_ct = generator_g(sample_mr)
to_mr = generator_f(sample_ct)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_mr, to_ct, sample_ct, to_mr]
title = ['FLAIR', 'To NCCT', 'NCCT', 'To FLAIR']

plt.subplot(2, 2, 1)
plt.title(title[0])
plt.imshow(sample_mr[0, daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
plt.subplot(2, 2, 2)
plt.title(title[1])
plt.imshow(sample_ct[0, daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
plt.subplot(2, 2, 3)
plt.title(title[2])
plt.imshow(to_ct[0, daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
plt.subplot(2, 2, 4)
plt.title(title[3])
plt.imshow(to_mr[0, daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)

plt.savefig(folder_prepend + "initial_state_gen.png")
plt.close()

plt.subplot(2, 1, 1)
plt.imshow(images_ct_flipped[30, 0, daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
plt.subplot(2, 1, 2)
plt.imshow(images_mr_flipped[30, 0, daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)

plt.savefig(folder_prepend + "initial_state_gen_flipped.png")
plt.close()


# Define loss functions

def feature_loss(mid_A, mid_B):
    return tf.reduce_mean(tf.abs(mid_A - mid_B))


def autoencoder_loss(input, reconstructed):
    loss1 = tf.reduce_mean(tf.abs(input - reconstructed))
    return LAMBDA * loss1


def attention_loss(dainput, mask):
    target_shape = dainput.shape 
    source_shape = mask.shape

    ratio = source_shape[1]/target_shape[1]

    sigma = 5.0 
    x = np.arange(-3,4,1)  
    y = np.arange(-3,4,1)
    z = np.arange(-3,4,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel_g = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    kernel_g = kernel_g / np.sum(kernel_g)

    kernel_gaussian = tf.constant(kernel_g, dtype=tf.float32)
    kernel_gaussian = kernel_gaussian[..., np.newaxis, np.newaxis]

    mask = tf.nn.conv3d(mask, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')
    mask = 0.75 + ((1 - mask) / 4)

    if ratio > 1:
        layer = tf.keras.layers.AveragePooling3D(pool_size=ratio)
        outputs = layer(mask)
    else:
        outputs = mask

    return dainput * outputs


def gauss_blur(mask):
    # sigma 5
    sigma = 2.0     # width of kernel
    x = np.arange(-3,4,1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-3,4,1)
    z = np.arange(-3,4,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel_g = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    kernel_g = kernel_g / np.sum(kernel_g)

    kernel_gaussian = tf.constant(kernel_g, dtype=tf.float32)
    kernel_gaussian = kernel_gaussian[..., np.newaxis, np.newaxis]

    return tf.nn.conv3d(mask, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def calc_lesion_loss(A, gnd_A, B, gnd_B):
    gnd_A_inv = 1.0 - gnd_A
    gnd_B_inv = 1.0 - gnd_B
    mask_bkg_A = tf.where(A > -0.9, 1.0, 0.0)
    mask_bkg_B = tf.where(B > -0.9, 1.0, 0.0)
    mask_bkg_A = mask_bkg_A * gnd_A_inv
    mask_bkg_B = mask_bkg_B * gnd_B_inv
    masked_bkg_A = A * mask_bkg_A
    masked_bkg_B = B * mask_bkg_B
    A_bkg_count = tf.reduce_sum(mask_bkg_A)
    B_bkg_count = tf.reduce_sum(mask_bkg_B)
    A_bkg_avg = tf.reduce_sum(masked_bkg_A) / (A_bkg_count + 0.0000000000001)
    B_bkg_avg = tf.reduce_sum(masked_bkg_B) / (B_bkg_count + 0.0000000000001)

    masked_A = A * gnd_A
    masked_B = B * gnd_B
    A_vox_count = tf.reduce_sum(gnd_A)
    B_vox_count = tf.reduce_sum(gnd_B)
    A_avg = tf.reduce_sum(masked_A) / (A_vox_count + 0.0000000000001)
    B_avg = tf.reduce_sum(masked_B) / (B_vox_count + 0.0000000000001)

    loss = tf.abs((A_avg/A_bkg_avg) - (B_avg/B_bkg_avg))

    return LAMBDA_C * loss


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


def GC_display(A, B, A_gnd, num, A_name, B_name):

    plt.subplot(1, 3, 1)
    plt.title(A_name)
    plt.imshow(A[0, daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(B_name)
    plt.imshow(B[0, daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
    plt.axis('off')

    # the difference map
    plt.subplot(1, 3, 3)
    plt.title("Difference map")
    diff = np.absolute((A[0, daslice, :, :, 0]+1) - (B[0, daslice, :, :, 0]+1))
    damax = np.nanmax(diff)
    plt.imshow(diff/damax, cmap='gray', vmin=0.0, vmax=1.0)
    plt.axis('off')

    plt.savefig(folder_prepend + 'cycle_' + A_name + "_" + num + '.jpg', dpi=300)
    plt.close()


def GC(A, B, A_gnd):

    A_gnd_inv = 1.0 - A_gnd

    # sigma 5
    sigma = 5.0     # width of kernel
    x = np.arange(-3,4,1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-3,4,1)
    z = np.arange(-3,4,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel_g = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    kernel_g = kernel_g / np.sum(kernel_g)

    kernel_gaussian = tf.constant(kernel_g, dtype=tf.float32)
    kernel_gaussian = kernel_gaussian[..., np.newaxis, np.newaxis]

    grad_op_A = A * A_gnd_inv
    grad_op_B = B * A_gnd_inv


    grad_op_A = tf.nn.conv3d(grad_op_A, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_B = tf.nn.conv3d(grad_op_B, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')

    grad_op_A = tf.nn.conv3d(grad_op_A, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_B = tf.nn.conv3d(grad_op_B, kernel_gaussian, strides=[1, 1, 1, 1, 1], padding='SAME')

    kernel_in_x = np.array([ [
        [ [ [-1] ],[ [0] ],[ [1] ] ],
        [ [ [-1] ],[ [0] ],[ [1] ] ],
        [ [ [-1] ],[ [0] ],[ [1] ] ],
        ],[
        [ [ [-1] ],[ [0] ],[ [1] ] ],
        [ [ [-2] ],[ [0] ],[ [2] ] ],
        [ [ [-1] ],[ [0] ],[ [1] ] ],
        ],[
        [ [ [-1] ],[ [0] ],[ [1] ] ],
        [ [ [-1] ],[ [0] ],[ [1] ] ],
        [ [ [-1] ],[ [0] ],[ [1] ] ],
        ]])
    kernel_x = tf.constant(kernel_in_x, dtype=tf.float32)
    #kernel_x = kernel_x[..., np.newaxis, np.newaxis]

    kernel_in_y = np.array([ [
        [ [ [1] ],[ [1] ],[ [1] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [-1] ],[ [-1] ],[ [-1] ] ],
        ],[
        [ [ [1] ],[ [2] ],[ [1] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [-1] ],[ [-2] ],[ [-1] ] ],
        ],[
        [ [ [1] ],[ [1] ],[ [1] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [-1] ],[ [-1] ],[ [-1] ] ],
        ]])
    kernel_y = tf.constant(kernel_in_y, dtype=tf.float32)
    #kernel_y = kernel_y[..., np.newaxis, np.newaxis]

    kernel_in_z = np.array([ [
        [ [ [1] ],[ [1] ],[ [1] ] ],
        [ [ [1] ],[ [2] ],[ [1] ] ],
        [ [ [1] ],[ [1] ],[ [1] ] ],
        ],[
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        ],[
        [ [ [-1] ],[ [-1] ],[ [-1] ] ],
        [ [ [-1] ],[ [-2] ],[ [-1] ] ],
        [ [ [-1] ],[ [-1] ],[ [-1] ] ],
        ]])
    kernel_z = tf.constant(kernel_in_z, dtype=tf.float32)
    #kernel_z = kernel_z[..., np.newaxis, np.newaxis]

    grad_op_x_A = tf.nn.conv3d(grad_op_A, kernel_x, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_x_B = tf.nn.conv3d(grad_op_B, kernel_x, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_y_A = tf.nn.conv3d(grad_op_A, kernel_y, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_y_B = tf.nn.conv3d(grad_op_B, kernel_y, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_z_A = tf.nn.conv3d(grad_op_A, kernel_z, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_z_B = tf.nn.conv3d(grad_op_B, kernel_z, strides=[1, 1, 1, 1, 1], padding='SAME')


    grad_op_x_A = grad_op_x_A * A_gnd_inv
    grad_op_x_B = grad_op_x_B * A_gnd_inv
    grad_op_y_A = grad_op_y_A * A_gnd_inv
    grad_op_y_B = grad_op_y_B * A_gnd_inv
    grad_op_z_A = grad_op_z_A * A_gnd_inv
    grad_op_z_B = grad_op_z_B * A_gnd_inv

    return (1/3)*(NCC(grad_op_x_A, grad_op_x_B)+NCC(grad_op_y_A, grad_op_y_B)+NCC(grad_op_z_A, grad_op_z_B))


def NCC(A, B):
    A_norm = A-tf.reduce_mean(A)
    B_norm = B-tf.reduce_mean(B)
    A_redux = tf.math.reduce_sum(A_norm*A_norm)
    B_redux = tf.math.reduce_sum(B_norm*B_norm)
    AB_root = tf.math.sqrt(A_redux*B_redux)
    
    return tf.math.reduce_sum(A_norm*B_norm)/(AB_root+0.0000000000000001)


generator_g_optimizer = tf.keras.optimizers.Adam(learning_rate_gen_g, beta_1=0.5, beta_2=0.999)
generator_f_optimizer = tf.keras.optimizers.Adam(learning_rate_gen_f, beta_1=0.5, beta_2=0.999)

discriminator_x_optimizer = tf.keras.optimizers.Adam(learning_rate_disc_f, beta_1=0.5, beta_2=0.999)
discriminator_y_optimizer = tf.keras.optimizers.Adam(learning_rate_disc_g, beta_1=0.5, beta_2=0.999)


def generate_images(model, test_input, label):
    
    prediction = model(test_input)
    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i][daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
        plt.axis('off')
    #plt.show()
    plt.savefig(folder_prepend + 'cyclegan_trained_' + label + '.png')
    plt.close()


# Define the training step 
@tf.function
def train_step(data):

    real_x, real_y, gnd_x, gnd_y = data

    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(tf.concat([real_x, gnd_x], axis=4), training=True) # , gnd_x
        disc_real_y = discriminator_y(tf.concat([real_y, gnd_y], axis=4), training=True) #y

        disc_fake_x = discriminator_x(tf.concat([fake_x, gnd_y], axis=4), training=True) #y
        disc_fake_y = discriminator_y(tf.concat([fake_y, gnd_x], axis=4), training=True) #x

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        # feature loss
        #total_feature_loss = feature_loss(mid_g([real_x])[0], mid_f([fake_y])[0]) + feature_loss(mid_f([real_y])[0], mid_g([fake_x])[0])
        # autoencoder loss
        #total_autoencoder_loss = autoencoder_loss(real_x, auto_g(real_x)) + autoencoder_loss(real_y, auto_f(real_y))
        # cycle loss
        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        structural_loss_g = LAMBDA_B * (1-GC(real_x, fake_y, gnd_x))
        structural_loss_f = LAMBDA_B * (1-GC(real_y, fake_x, gnd_y))

        #lesion_loss_g = calc_lesion_loss(fake_y, gnd_x, real_y, gnd_y)
        #lesion_loss_f = calc_lesion_loss(fake_x, gnd_y, real_x, gnd_x)
        
        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + identity_loss(real_y, same_y) + total_cycle_loss + structural_loss_g #+ total_autoencoder_loss + total_feature_loss
        total_gen_f_loss = gen_f_loss + identity_loss(real_x, same_x) + total_cycle_loss + structural_loss_f #+ total_autoencoder_loss + total_feature_loss

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                        generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                        generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                            discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                            discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                            generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                            generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

    return total_cycle_loss, (structural_loss_g + structural_loss_f), disc_x_loss, disc_y_loss, gen_g_loss, gen_f_loss

# transpose - pair images with their ground truths

if len(val_images_mr) <= len(val_images_ct):
    test_lim = len(val_images_mr)
else:
    test_lim = len(val_images_ct)

# Validation step

def evaluate_loss():
    accum = []
    accum_feature = []
    accum_struct = []
    accum_disc = []
    accum_gen = []
    for nn in range(test_lim):

        real_x, real_y, gnd_x, gnd_y = val_images_mr[nn], val_images_ct[nn], val_gnd_images_mr[nn], val_gnd_images_ct[nn]

        fake_y = generator_g(real_x, training=False)
        cycled_x = generator_f(fake_y, training=False)

        fake_x = generator_f(real_y, training=False)
        cycled_y = generator_g(fake_x, training=False)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(tf.concat([real_x, gnd_x], axis=4), training=False)
        disc_real_y = discriminator_y(tf.concat([real_y, gnd_y], axis=4), training=False)

        disc_fake_x = discriminator_x(tf.concat([fake_x, gnd_y], axis=4), training=False)
        disc_fake_y = discriminator_y(tf.concat([fake_y, gnd_x], axis=4), training=False)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        structural_loss_g = LAMBDA_B * (1-GC(real_x, fake_y, gnd_x))
        structural_loss_f = LAMBDA_B * (1-GC(real_y, fake_x, gnd_y))
        
        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y) + structural_loss_g #+ total_feature_loss
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x) + structural_loss_f #+ total_feature_loss

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

        total_cycle_loss, (structural_loss_g + structural_loss_f), (disc_x_loss + disc_y_loss), (gen_g_loss + gen_f_loss)

        accum.append(total_cycle_loss)
        accum_struct.append(structural_loss_g + structural_loss_f)
        accum_disc.append(disc_x_loss + disc_y_loss)
        accum_gen.append(gen_g_loss + gen_f_loss)

    accum = np.array(accum)
    accum = accum.mean()

    accum_struct = np.array(accum_struct)
    accum_struct = accum_struct.mean()

    accum_disc = np.array(accum_disc)
    accum_disc = accum_disc.mean()

    accum_gen = np.array(accum_gen)
    accum_gen = accum_gen.mean()

    return accum, accum_struct, accum_disc, accum_gen


process_m = psutil.Process(os.getpid())
print("pre-elastic - RAM %.2f Gb" % (process_m.memory_info().rss/1000/1000/1000))

# Generate elastic augmentations MR.

elastic_gen = ea.WarpEngine(0)

if not os.path.exists(scratch_path+'np_datasets/fold_'+str(fold)+'/images_mr_anup_elastic_aug.npy'):

    aug_cycles = 2

    # elastic mrs
    dashape = images_mr.shape
    print(dashape)
    dashape = list(dashape)
    dashape[0] = int(dashape[0] * aug_cycles)
    dashape = tuple(dashape)
    print(dashape)

    images_mr_aug = np.empty(dashape, dtype='float32')
    images_mr_aug_flipped = np.empty(dashape, dtype='float32')
    gnd_images_mr_aug = np.empty(dashape, dtype='float32')
    gnd_images_mr_aug_flipped = np.empty(dashape, dtype='float32')


    for pr in range(len(images_mr)):
        image_x = images_mr[pr]
        image_x_gnd = gnd_images_mr[pr]
        for q in range(aug_cycles):
            daindex = pr+(q*len(images_mr))
            print(daindex)
            image_x_out = np.array(elastic_gen.generate(image_x[0, :, :, :, 0], image_x_gnd[0, :, :, :, 0], 1))
            images_mr_aug[daindex] = image_x_out[0]
            gnd_images_mr_aug[daindex] = image_x_out[1]
            images_mr_aug_flipped[daindex] = np.flip(image_x_out[0, 0, :, :, :, 0], 2)[np.newaxis, ..., np.newaxis]
            gnd_images_mr_aug_flipped[daindex] = np.flip(image_x_out[1, 0, :, :, :, 0], 2)[np.newaxis, ..., np.newaxis]
            
        process_m = psutil.Process(os.getpid())
        print("MR augmentation "+str(pr)+" complete - RAM %.2f Gb" % (process_m.memory_info().rss/1000/1000/1000))

    np.save(scratch_path+"np_datasets/fold_"+str(fold)+"/images_mr_anup_elastic_aug.npy", images_mr_aug)
    np.save(scratch_path+"np_datasets/fold_"+str(fold)+"/images_mr_anup_elastic_aug_flipped.npy", images_mr_aug_flipped)
    np.save(scratch_path+"np_datasets/fold_"+str(fold)+"/images_mr_anup_elastic_aug_gnd.npy", gnd_images_mr_aug)
    np.save(scratch_path+"np_datasets/fold_"+str(fold)+"/images_mr_anup_elastic_aug_flipped_gnd.npy", gnd_images_mr_aug_flipped)

else:

    images_mr_aug = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/images_mr_anup_elastic_aug.npy')
    images_mr_aug_flipped = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/images_mr_anup_elastic_aug_flipped.npy')
    gnd_images_mr_aug = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/images_mr_anup_elastic_aug_gnd.npy')
    gnd_images_mr_aug_flipped = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/images_mr_anup_elastic_aug_flipped_gnd.npy')

# Generate elastic augmentations CT.

if not os.path.exists(scratch_path+'np_datasets/fold_'+str(fold)+'/images_ct_anup_elastic_aug.npy'):
    
    aug_cycles = 2

    # elastic cts
    dashape = images_ct.shape
    dashape = list(dashape)
    dashape[0] = int(dashape[0] * aug_cycles)
    dashape = tuple(dashape)

    images_ct_aug = np.empty(dashape, dtype='float32')
    images_ct_aug_flipped = np.empty(dashape, dtype='float32')
    gnd_images_ct_aug = np.empty(dashape, dtype='float32')
    gnd_images_ct_aug_flipped = np.empty(dashape, dtype='float32')
 

    for pr in range(len(images_ct)):
        image_y = images_ct[pr]
        image_y_gnd = gnd_images_ct[pr]
        for q in range(aug_cycles):
            daindex = pr+(q*len(images_ct))
            print(daindex)
            image_y_out = np.array(elastic_gen.generate(image_y[0, :, :, :, 0], image_y_gnd[0, :, :, :, 0], 1))
            images_ct_aug[daindex] = image_y_out[0]
            gnd_images_ct_aug[daindex] = image_y_out[1]
            images_ct_aug_flipped[daindex] = np.flip(image_y_out[0, 0, :, :, :, 0], 2)[np.newaxis, ..., np.newaxis]
            gnd_images_ct_aug_flipped[daindex] = np.flip(image_y_out[1, 0, :, :, :, 0], 2)[np.newaxis, ..., np.newaxis]
            
        process_m = psutil.Process(os.getpid())
        print("CT augmentation "+str(pr)+" complete - RAM %.2f Gb" % (process_m.memory_info().rss/1000/1000/1000))

    np.save(scratch_path+"np_datasets/fold_"+str(fold)+"/images_ct_anup_elastic_aug.npy", images_ct_aug)
    np.save(scratch_path+"np_datasets/fold_"+str(fold)+"/images_ct_anup_elastic_aug_flipped.npy", images_ct_aug_flipped)
    np.save(scratch_path+"np_datasets/fold_"+str(fold)+"/images_ct_anup_elastic_aug_gnd.npy", gnd_images_ct_aug)
    np.save(scratch_path+"np_datasets/fold_"+str(fold)+"/images_ct_anup_elastic_aug_flipped_gnd.npy", gnd_images_ct_aug_flipped)


else:

    images_ct_aug = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/images_ct_anup_elastic_aug.npy')
    images_ct_aug_flipped = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/images_ct_anup_elastic_aug_flipped.npy')
    gnd_images_ct_aug = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/images_ct_anup_elastic_aug_gnd.npy')
    gnd_images_ct_aug_flipped = np.load(scratch_path+'np_datasets/fold_'+str(fold)+'/images_ct_anup_elastic_aug_flipped_gnd.npy')



process_m = psutil.Process(os.getpid())
print("post-elastic - RAM %.2f Gb" % (process_m.memory_info().rss/1000/1000/1000))


# Create loss arrays
loss = []
val_loss = []
struc_loss = []
val_struc_loss = []
gen_loss_g = []
gen_loss_f = []
val_gen_loss = []
disc_loss_x = []
disc_loss_y = []
val_disc_loss = []


# Training model begins.

BUFFER_SIZE = 40

for epoch in range(EPOCHS):

    start = time.time()
    daloss = []
    dbloss = []
    ddxloss = []
    ddyloss = []
    dggloss = []
    dgfloss = []
    mrs_idxs = np.array(range(len(images_mr)))
    cts_idxs = np.array(range(len(images_ct)))
    np.random.shuffle(mrs_idxs)
    np.random.shuffle(cts_idxs)

    for pr in range(BUFFER_SIZE): 

        pr_mr = mrs_idxs[pr]
        pr_ct = cts_idxs[pr]

        aloss, bloss, dxloss, dyloss, ggloss, gfloss = train_step([images_mr[pr_mr], images_ct[pr_ct], gnd_images_mr[pr_mr], gnd_images_ct[pr_ct]])
        daloss.append(aloss)
        dbloss.append(bloss)
        ddxloss.append(dxloss)
        ddyloss.append(dyloss)
        dggloss.append(ggloss)
        dgfloss.append(gfloss)

        aloss, bloss, dxloss, dyloss, ggloss, gfloss = train_step([images_mr_flipped[pr_mr], images_ct_flipped[pr_ct], gnd_images_mr_flipped[pr_mr], gnd_images_ct_flipped[pr_ct]])
        daloss.append(aloss)
        dbloss.append(bloss)
        ddxloss.append(dxloss)
        ddyloss.append(dyloss)
        dggloss.append(ggloss)
        dgfloss.append(gfloss)

        
        aloss, bloss, dxloss, dyloss, ggloss, gfloss = train_step([images_mr_aug[pr_mr], images_ct_aug[pr_ct], gnd_images_mr_aug[pr_mr], gnd_images_ct_aug[pr_ct]])
        daloss.append(aloss)
        dbloss.append(bloss)
        ddxloss.append(dxloss)
        ddyloss.append(dyloss)
        dggloss.append(ggloss)
        dgfloss.append(gfloss)

        aloss, bloss, dxloss, dyloss, ggloss, gfloss = train_step([images_mr_aug_flipped[pr_mr], images_ct_aug_flipped[pr_ct], gnd_images_mr_aug_flipped[pr_mr], gnd_images_ct_aug_flipped[pr_ct]])
        daloss.append(aloss)
        dbloss.append(bloss)
        ddxloss.append(dxloss)
        ddyloss.append(dyloss)
        dggloss.append(ggloss)
        dgfloss.append(gfloss)
        
    
    # Display sample images translated with updated version of the model. 
    GC_display(sample_mr, generator_g(sample_mr), sample_mr_gnd, str(epoch), "FLAIR", "NCCT")
    GC_display(sample_ct, generator_f(sample_ct), sample_ct_gnd, str(epoch), "NCCT", "FLAIR")

    GC_display(sample_mr_test, generator_g(sample_mr_test), sample_mr_gnd_test, str(epoch), "FLAIR_val", "NCCT_val")
    GC_display(sample_ct_test, generator_f(sample_ct_test), sample_ct_gnd_test, str(epoch), "NCCT_val", "FLAIR_val")


    plt.figure(figsize=(8, 8))

    plt.subplot(121)
    plt.title('real NCCT')
    plt.imshow(discriminator_y(tf.concat([sample_ct, sample_ct_gnd], axis=4))[0, 3, :, :, -1], cmap='RdBu_r')

    plt.subplot(122)
    plt.title('real MR')
    plt.imshow(discriminator_x(tf.concat([sample_mr, sample_mr_gnd], axis=4))[0, 3, :, :, -1], cmap='RdBu_r')

    plt.savefig(folder_prepend + "disc_real_"+str(epoch)+".png")
    plt.close()

    plt.figure(figsize=(8, 8))

    plt.subplot(121)
    plt.title('fake MR')
    plt.imshow(discriminator_x(tf.concat([generator_f(sample_ct), sample_ct_gnd], axis=4))[0, 3, :, :, -1], cmap='RdBu_r')

    plt.subplot(122)
    plt.title('fake NCCT')
    plt.imshow(discriminator_y(tf.concat([generator_g(sample_mr), sample_mr_gnd], axis=4))[0, 3, :, :, -1], cmap='RdBu_r')

    plt.savefig(folder_prepend + "disc_fake_"+str(epoch)+".png")
    plt.close()

    
    print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Append measured loss to its corresponding array.

    daloss = np.array(daloss)
    daloss = daloss.mean()
    dbloss = np.array(dbloss)
    dbloss = dbloss.mean()
    dggloss = np.array(dggloss)
    dggloss = dggloss.mean()
    dgfloss = np.array(dgfloss)
    dgfloss = dgfloss.mean()
    ddxloss = np.array(ddxloss)
    ddxloss = ddxloss.mean()
    ddyloss = np.array(ddyloss)
    ddyloss = ddyloss.mean()
    daloss_test, dbloss_test, ddloss_test, dgloss_test = evaluate_loss()
    print('Train loss: ' + str(daloss) + ' - Struc loss: ' + str(dbloss) + ' - Disc loss x: ' + str(ddxloss) + ' - Disc loss y: ' + str(ddyloss) + ' - Gen loss g: ' + str(dggloss)+ ' - Gen loss f: ' + str(dgfloss))
    print('Test loss: ' + str(daloss_test) + ' - Struc loss: ' + str(dbloss_test) + ' - Disc loss: ' + str(ddloss_test) + ' - Gen loss: ' + str(dgloss_test))

    loss.append(daloss)
    val_loss.append(daloss_test)
    struc_loss.append(dbloss)
    val_struc_loss.append(dbloss_test)
    disc_loss_x.append(ddxloss)
    disc_loss_y.append(ddyloss)
    val_disc_loss.append(ddloss_test)
    gen_loss_g.append(dggloss)  
    gen_loss_f.append(dgfloss)  
    val_gen_loss.append(dgloss_test)

    # Every 20 epochs update the plots of losses.

    if (epoch+1)%20 == 0:

        epochss = range(epoch+1)

        plt.figure()
        plt.plot(epochss, loss, 'bo', label='Training consistency loss', linewidth=0.5)
        plt.plot(epochss, val_loss, 'r', label='Validation consistency loss', linewidth=0.5)
        plt.title('loss', fontsize=10)
        plt.legend()
        plt.savefig(folder_prepend + 'loss_curve.png')

        plt.figure()
        plt.plot(epochss, struc_loss, 'bo', label='Training structural loss', linewidth=0.5)
        plt.plot(epochss, val_struc_loss, 'r', label='Validation loss', linewidth=0.5)
        plt.title('loss', fontsize=10)
        plt.legend()
        plt.savefig(folder_prepend + 'loss_curve_struc.png')

        plt.figure()
        plt.plot(epochss, disc_loss_y, 'bo', label='Discriminator loss y', linewidth=0.5)
        plt.plot(epochss, disc_loss_x, 'go', label='Discriminator loss x', linewidth=0.5)
        plt.plot(epochss, val_disc_loss, 'r', label='Validation loss', linewidth=0.5)
        plt.title('loss', fontsize=10)
        plt.legend()
        plt.savefig(folder_prepend + 'loss_curve_disc.png')

        plt.figure()
        plt.plot(epochss, gen_loss_g, 'bo', label='Generator loss g', linewidth=0.5)
        plt.plot(epochss, gen_loss_f, 'go', label='Generator loss f', linewidth=0.5)
        plt.plot(epochss, val_gen_loss, 'r', label='Validation loss', linewidth=0.5)
        plt.title('loss', fontsize=10)
        plt.legend()
        plt.savefig(folder_prepend + 'loss_curve_gen.png')

        # Save checkpoint of weights

        #generator_g.save_weights(folder_prepend + 'checkpoints/generator_mr_to_ct_'+str(epoch+1)+'.h5')
        #generator_f.save_weights(folder_prepend + 'checkpoints/generator_ct_to_mr_'+str(epoch+1)+'.h5')
        #discriminator_x.save_weights(folder_prepend + 'checkpoints/discriminator_mr_to_ct_'+str(epoch+1)+'.h5')
        #discriminator_y.save_weights(folder_prepend + 'checkpoints/discriminator_ct_to_mr_'+str(epoch+1)+'.h5')


# Plot all loss curves. Save final models.
epochss = range(EPOCHS)

plt.figure()
plt.plot(epochss, loss, 'bo', label='Training consistency loss', linewidth=0.5)
plt.plot(epochss, val_loss, 'r', label='Validation consistency loss', linewidth=0.5)
plt.title('loss', fontsize=10)
plt.legend()
plt.savefig(folder_prepend + 'loss_curve.png')

plt.figure()
plt.plot(epochss, struc_loss, 'bo', label='Training structural loss', linewidth=0.5)
plt.plot(epochss, val_struc_loss, 'r', label='Validation loss', linewidth=0.5)
plt.title('loss', fontsize=10)
plt.legend()
plt.savefig(folder_prepend + 'loss_curve_struc.png')

plt.figure()
plt.plot(epochss, disc_loss_y, 'bo', label='Discriminator loss y', linewidth=0.5)
plt.plot(epochss, disc_loss_x, 'go', label='Discriminator loss x', linewidth=0.5)
plt.plot(epochss, val_disc_loss, 'r', label='Validation loss', linewidth=0.5)
plt.title('loss', fontsize=10)
plt.legend()
plt.savefig(folder_prepend + 'loss_curve_disc.png')

plt.figure()
plt.plot(epochss, gen_loss_g, 'bo', label='Generator loss g', linewidth=0.5)
plt.plot(epochss, gen_loss_f, 'go', label='Generator loss f', linewidth=0.5)
plt.plot(epochss, val_gen_loss, 'r', label='Validation loss', linewidth=0.5)
plt.title('loss', fontsize=10)
plt.legend()
plt.savefig(folder_prepend + 'loss_curve_gen.png')

generator_g.save_weights(folder_prepend + 'generator_mr_to_ct.h5')
generator_f.save_weights(folder_prepend + 'generator_ct_to_mr.h5')

discriminator_x.save_weights(folder_prepend + 'discriminator_mr_to_ct.h5')
discriminator_y.save_weights(folder_prepend + 'discriminator_ct_to_mr.h5')