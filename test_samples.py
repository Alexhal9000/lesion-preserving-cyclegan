import tensorflow as tf
import tensorflow_datasets as tfds
import patientkeyinfo as pki
import SimpleITK as sitk
import pix2pix3D_backup as pix2pix
import pix2pix3D_vanilla as pix2pix_van
import numpy as np
import scipy
from scipy import stats
from scipy import ndimage
from skimage.segmentation import flood, flood_fill
import psutil
import gc
import seaborn as sns
import pandas as pd

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

# load scans NCCT and FLAIR

scale_factor = 1
target_scale = 192
size_offset = 0

images_mr_names = []
images_ct_names = []
images_mr_names_flipped = []
images_ct_names_flipped = []
val_images_mr_names = []
val_images_ct_names = []
test_images_mr_names = []
test_images_ct_names = []

folder_prepend = "./eval_4_all/"

daslice = 32


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


def cohens_d(a, b):
    return (np.mean(a) - np.mean(b)) / (np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2))

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    
    """
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


scratch_path = "../../../../../scratch/alexgumu/np_datasets/fold_4/"


if not os.path.exists(scratch_path+'images_mr_anup.npy'):

    # source folder, csv keys, seed, num of train samples
    patients_mr = pki.PatientKeyInfo('./data_to_interpolate/stroke_project_data/images_mr/*.nii.gz', 'none', 10, -1)
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
    patients_ct = pki.PatientKeyInfo('./data_to_interpolate/stroke_project_data/images_ct/*.nii.gz', 'none', 10, -1)
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
    patients_mr_lesion = pki.PatientKeyInfo('./data_to_interpolate/stroke_project_data/grounds_mr/*.nii.gz', 'none', 10, -1)
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
    patients_ct_lesion = pki.PatientKeyInfo('./data_to_interpolate/stroke_project_data/grounds_ct/*.nii.gz', 'none', 10, -1)
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



    """
    # atlas mr
    itkImage_mr_atlas = load_vol_ITK("./miplab-flair_asym_brain.nii.gz")
    Xmr_atlas = sitk.GetArrayFromImage(itkImage_mr_atlas).astype('float32')

    if scale_factor != 1:
        Xmr_atlas = scipy.ndimage.interpolation.zoom(Xmr_atlas, scale_factor, order=1)
    Xmr_atlas_resized = np.zeros(np.array([target_scale, target_scale, target_scale]) + size_offset, np.float32)
    Xmr_atlas_resized[:Xmr_atlas.shape[0], :Xmr_atlas.shape[1], :Xmr_atlas.shape[2]] = Xmr_atlas
    Xmr_atlas_resized = np.interp(Xmr_atlas_resized, (Xmr_atlas_resized.min(), Xmr_atlas_resized.max()), (-1, 1))
    #Xmr_atlas_resized = Xmr_atlas_resized[np.newaxis, ..., np.newaxis]
    """

    # train mr
    first_flag = True
    for idx in range(len(train_vol_names_mr)):

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

        #Xmr_resized = hist_match(Xmr_resized, Xmr_atlas_resized)
        Xmr_flipped = np.copy(Xmr_resized)
        
        # ground truth
        gnd_Xmr = gnd_Xmr_vol
        
        if scale_factor != 1:
            gnd_Xmr = scipy.ndimage.interpolation.zoom(gnd_Xmr, scale_factor, order=0)

        gnd_Xmr_resized = np.zeros(np.array([64, 160, 160]) + size_offset, np.float32)
        gnd_Xmr_resized = gnd_Xmr[80:144, 16:176, 16:176]
        
        gnd_Xmr_resized = np.where(gnd_Xmr_resized>0.0, 1.0, 0.0)

        """
        # snug fit gnds
        count_vox_x = np.sum(gnd_Xmr_resized)
        masked_gnd_x = ndimage.gaussian_filter(Xmr_resized+1.0, sigma=5) * gnd_Xmr_resized
        masked_mean_x = np.sum(masked_gnd_x)/(count_vox_x+0.000000001)
        masked_gnd_square_x = ((masked_gnd_x - masked_mean_x) * gnd_Xmr_resized) * ((masked_gnd_x - masked_mean_x) * gnd_Xmr_resized)
        masked_std_x = np.sqrt(np.sum(masked_gnd_square_x)/(count_vox_x+0.000000001))
        gnd_Xmr_resized = np.where(masked_gnd_x>=(masked_mean_x-(masked_std_x)), 1.0, 0.0)
        """

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



    # train ct
    first_flag = True
    for idx in range(len(train_vol_names_ct)):

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

        """
        # snug fit gnds
        count_vox_y = np.sum(gnd_Xct_resized)
        masked_gnd_y = ndimage.gaussian_filter(2.0-(Xct_resized+1.0), sigma=5) * gnd_Xct_resized
        masked_mean_y = np.sum(masked_gnd_y)/(count_vox_y+0.000000001)
        masked_gnd_square_y = ((masked_gnd_y - masked_mean_y) * gnd_Xct_resized) * ((masked_gnd_y - masked_mean_y) * gnd_Xct_resized)
        masked_std_y = np.sqrt(np.sum(masked_gnd_square_y)/(count_vox_y+0.000000001))
        gnd_Xct_resized = np.where(masked_gnd_y>=(masked_mean_y-(masked_std_y)), 1.0, 0.0)
        """            

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


    # val mr
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

        """
        # snug fit gnds
        count_vox_x = np.sum(gnd_Xmr_resized)
        masked_gnd_x = ndimage.gaussian_filter(Xmr_resized+1.0, sigma=5) * gnd_Xmr_resized
        masked_mean_x = np.sum(masked_gnd_x)/(count_vox_x+0.000000001)
        masked_gnd_square_x = ((masked_gnd_x - masked_mean_x) * gnd_Xmr_resized) * ((masked_gnd_x - masked_mean_x) * gnd_Xmr_resized)
        masked_std_x = np.sqrt(np.sum(masked_gnd_square_x)/(count_vox_x+0.000000001))
        gnd_Xmr_resized = np.where(masked_gnd_x>=(masked_mean_x-(masked_std_x)), 1.0, 0.0)
        """

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




    # val ct
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

        """
        # snug fit gnds
        count_vox_y = np.sum(gnd_Xct_resized)
        masked_gnd_y = ndimage.gaussian_filter(2.0-(Xct_resized+1.0), sigma=5) * gnd_Xct_resized
        masked_mean_y = np.sum(masked_gnd_y)/(count_vox_y+0.000000001)
        masked_gnd_square_y = ((masked_gnd_y - masked_mean_y) * gnd_Xct_resized) * ((masked_gnd_y - masked_mean_y) * gnd_Xct_resized)
        masked_std_y = np.sqrt(np.sum(masked_gnd_square_y)/(count_vox_y+0.000000001))
        gnd_Xct_resized = np.where(masked_gnd_y>=(masked_mean_y-(masked_std_y)), 1.0, 0.0)
        """            

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


    # test mr
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

        """
        # snug fit gnds
        count_vox_x = np.sum(gnd_Xmr_resized)
        masked_gnd_x = ndimage.gaussian_filter(Xmr_resized+1.0, sigma=5) * gnd_Xmr_resized
        masked_mean_x = np.sum(masked_gnd_x)/(count_vox_x+0.000000001)
        masked_gnd_square_x = ((masked_gnd_x - masked_mean_x) * gnd_Xmr_resized) * ((masked_gnd_x - masked_mean_x) * gnd_Xmr_resized)
        masked_std_x = np.sqrt(np.sum(masked_gnd_square_x)/(count_vox_x+0.000000001))
        gnd_Xmr_resized = np.where(masked_gnd_x>=(masked_mean_x-(masked_std_x)), 1.0, 0.0)
        """

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




    # test ct
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

        """
        # snug fit gnds
        count_vox_y = np.sum(gnd_Xct_resized)
        masked_gnd_y = ndimage.gaussian_filter(2.0-(Xct_resized+1.0), sigma=5) * gnd_Xct_resized
        masked_mean_y = np.sum(masked_gnd_y)/(count_vox_y+0.000000001)
        masked_gnd_square_y = ((masked_gnd_y - masked_mean_y) * gnd_Xct_resized) * ((masked_gnd_y - masked_mean_y) * gnd_Xct_resized)
        masked_std_y = np.sqrt(np.sum(masked_gnd_square_y)/(count_vox_y+0.000000001))
        gnd_Xct_resized = np.where(masked_gnd_y>=(masked_mean_y-(masked_std_y)), 1.0, 0.0)
        """            

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

    """
    images_mr = np.array(images_mr)
    images_ct = np.array(images_ct)
    gnd_images_mr = np.array(gnd_images_mr)
    gnd_images_ct = np.array(gnd_images_ct)


    images_mr_flipped = np.array(images_mr_flipped)
    images_ct_flipped = np.array(images_ct_flipped)
    gnd_images_mr_flipped = np.array(gnd_images_mr_flipped)
    gnd_images_ct_flipped = np.array(gnd_images_ct_flipped)


    back_images_mr = np.array(back_images_mr)
    back_images_ct = np.array(back_images_ct)
    back_images_mr_flipped = np.array(back_images_mr_flipped)
    back_images_ct_flipped = np.array(back_images_ct_flipped)
    val_back_images_mr = np.array(val_back_images_mr)
    val_back_images_ct = np.array(val_back_images_ct)

    val_images_mr = np.array(val_images_mr)
    val_images_ct = np.array(val_images_ct)
    val_gnd_images_mr = np.array(val_gnd_images_mr)
    val_gnd_images_ct = np.array(val_gnd_images_ct)

    test_images_mr = np.array(test_images_mr)
    test_images_ct = np.array(test_images_ct)
    test_gnd_images_mr = np.array(test_gnd_images_mr)
    test_gnd_images_ct = np.array(test_gnd_images_ct)
    """

    # inspect images
    """

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
    """


    with open(scratch_path+'images_mr_anup.npy', 'wb') as f:
        np.save(f, images_mr)
    with open(scratch_path+'images_ct_anup.npy', 'wb') as f:
        np.save(f, images_ct)
    with open(scratch_path+'gnd_images_mr_anup.npy', 'wb') as f:
        np.save(f, gnd_images_mr)
    with open(scratch_path+'gnd_images_ct_anup.npy', 'wb') as f:
        np.save(f, gnd_images_ct)

    with open(scratch_path+'images_mr_flipped_anup.npy', 'wb') as f:
        np.save(f, images_mr_flipped)
    with open(scratch_path+'images_ct_flipped_anup.npy', 'wb') as f:
        np.save(f, images_ct_flipped)
    with open(scratch_path+'gnd_images_mr_flipped_anup.npy', 'wb') as f:
        np.save(f, gnd_images_mr_flipped)
    with open(scratch_path+'gnd_images_ct_flipped_anup.npy', 'wb') as f:
        np.save(f, gnd_images_ct_flipped)

    with open(scratch_path+'val_images_mr_anup.npy', 'wb') as f:
        np.save(f, val_images_mr)
    with open(scratch_path+'val_images_ct_anup.npy', 'wb') as f:
        np.save(f, val_images_ct)
    with open(scratch_path+'val_gnd_images_mr_anup.npy', 'wb') as f:
        np.save(f, val_gnd_images_mr)
    with open(scratch_path+'val_gnd_images_ct_anup.npy', 'wb') as f:
        np.save(f, val_gnd_images_ct)

    with open(scratch_path+'test_images_mr_anup.npy', 'wb') as f:
        np.save(f, test_images_mr)
    with open(scratch_path+'test_images_ct_anup.npy', 'wb') as f:
        np.save(f, test_images_ct)
    with open(scratch_path+'test_gnd_images_mr_anup.npy', 'wb') as f:
        np.save(f, test_gnd_images_mr)
    with open(scratch_path+'test_gnd_images_ct_anup.npy', 'wb') as f:
        np.save(f, test_gnd_images_ct)

    with open(scratch_path+'back_images_mr_anup.npy', 'wb') as f:
        np.save(f, back_images_mr)
    with open(scratch_path+'val_back_images_mr_anup.npy', 'wb') as f:
        np.save(f, val_back_images_mr)
    with open(scratch_path+'back_images_ct_anup.npy', 'wb') as f:
        np.save(f, back_images_ct)
    with open(scratch_path+'val_back_images_ct_anup.npy', 'wb') as f:
        np.save(f, val_back_images_ct)
    with open(scratch_path+'back_images_mr_anup_flipped.npy', 'wb') as f:
        np.save(f, back_images_mr_flipped)
    with open(scratch_path+'back_images_ct_anup_flipped.npy', 'wb') as f:
        np.save(f, back_images_ct_flipped)


else:

    images_mr = np.load(scratch_path+'images_mr_anup.npy')
    images_ct = np.load(scratch_path+'images_ct_anup.npy')
    gnd_images_mr = np.load(scratch_path+'gnd_images_mr_anup.npy')
    gnd_images_ct = np.load(scratch_path+'gnd_images_ct_anup.npy')

    images_mr_flipped = np.load(scratch_path+'images_mr_flipped_anup.npy')
    images_ct_flipped = np.load(scratch_path+'images_ct_flipped_anup.npy')
    gnd_images_mr_flipped = np.load(scratch_path+'gnd_images_mr_flipped_anup.npy')
    gnd_images_ct_flipped = np.load(scratch_path+'gnd_images_ct_flipped_anup.npy')

    val_images_mr = np.load(scratch_path+'val_images_mr_anup.npy')
    val_images_ct = np.load(scratch_path+'val_images_ct_anup.npy')
    val_gnd_images_mr = np.load(scratch_path+'val_gnd_images_mr_anup.npy')
    val_gnd_images_ct = np.load(scratch_path+'val_gnd_images_ct_anup.npy')

    
    test_images_mr = np.load(scratch_path+'test_images_mr_anup.npy')
    test_images_ct = np.load(scratch_path+'test_images_ct_anup.npy')
    test_gnd_images_mr = np.load(scratch_path+'test_gnd_images_mr_anup.npy')
    test_gnd_images_ct = np.load(scratch_path+'test_gnd_images_ct_anup.npy')    

    back_images_mr = np.load(scratch_path+'back_images_mr_anup.npy')
    back_images_ct = np.load(scratch_path+'back_images_ct_anup.npy')
    back_images_mr_flipped = np.load(scratch_path+'back_images_mr_anup_flipped.npy')
    back_images_ct_flipped = np.load(scratch_path+'back_images_ct_anup_flipped.npy')
    val_back_images_mr = np.load(scratch_path+'val_back_images_mr_anup.npy')
    val_back_images_ct = np.load(scratch_path+'val_back_images_ct_anup.npy')
  


print(images_ct.shape)
print(val_images_ct.shape)
print(test_images_ct.shape)
print(back_images_ct.shape)
print(back_images_ct_flipped.shape)

print(images_mr.shape)
print(val_images_mr.shape)
print(test_images_mr.shape)
print(back_images_mr.shape)
print(back_images_mr_flipped.shape)

BATCH_SIZE = 1
IMG_WIDTH = 192
IMG_HEIGHT = 192
IMG_DEPTH = 192


# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    max_val = tf.math.reduce_max(image)
    image = (image / max_val) - 1
    return image


def preprocess_image_train(image, label):
    image = normalize(image)
    return image

def preprocess_image_test(image, label):
    image = tf.image.resize(image, [160, 160],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = normalize(image)
    return image


#print(train_horses)

#print(iter(train_horses))

#sample_horse = np.stack(list(iter(train_horses)))[10]
#sample_zebra = next(iter(train_zebras))




OUTPUT_CHANNELS = 1


"""
plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real NCCT?')
plt.imshow(discriminator_y(sample_ct)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real MR?')
plt.imshow(discriminator_x(sample_mr)[0, ..., -1], cmap='RdBu_r')

#plt.show()
plt.savefig(folder_prepend + "initial_state_disc.png")
plt.close()
"""


LAMBDA = 4 #8
LAMBDA_B = 0.2 #0.4
LAMBDA_C = 0.1
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


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

    grad_op_x_A = tf.nn.conv3d(grad_op_A, kernel_x, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_x_B = tf.nn.conv3d(grad_op_B, kernel_x, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_y_A = tf.nn.conv3d(grad_op_A, kernel_y, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_y_B = tf.nn.conv3d(grad_op_B, kernel_y, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_z_A = tf.nn.conv3d(grad_op_A, kernel_z, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_z_B = tf.nn.conv3d(grad_op_B, kernel_z, strides=[1, 1, 1, 1, 1], padding='SAME')

    # inverse
    """

    kernel_in_x_inv = np.array([ 
        [ [ [1] ],[ [0] ],[ [-1] ] ],
        [ [ [2] ],[ [0] ],[ [-2] ] ],
        [ [ [1] ],[ [0] ],[ [-1] ] ],
        ])
    kernel_x_inv = tf.constant(kernel_in_x_inv, dtype=tf.float32)

    kernel_in_y_inv = np.array([ 
        [ [ [-1] ],[ [-2] ],[ [-1] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [1] ],[ [2] ],[ [1] ] ],
        ])
    kernel_y_inv = tf.constant(kernel_in_y_inv, dtype=tf.float32)

    grad_op_x_B_inv = tf.nn.conv3d(grad_op_B, kernel_x_inv, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_y_B_inv = tf.nn.conv3d(grad_op_B, kernel_y_inv, strides=[1, 1, 1, 1, 1], padding='SAME')

    grad_op_x_B = (grad_op_x_B * A_gnd_inv) + (grad_op_x_B_inv * A_gnd)
    grad_op_y_B = (grad_op_y_B * A_gnd_inv) + (grad_op_y_B_inv * A_gnd)
    """

    grad_op_x_A = grad_op_x_A * A_gnd_inv
    grad_op_x_B = grad_op_x_B * A_gnd_inv
    grad_op_y_A = grad_op_y_A * A_gnd_inv
    grad_op_y_B = grad_op_y_B * A_gnd_inv
    grad_op_z_A = grad_op_z_A * A_gnd_inv
    grad_op_z_B = grad_op_z_B * A_gnd_inv
    
    plt.subplot(2, 4, 1)
    plt.title(A_name)
    plt.imshow(A[0, daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.title("Lesion")
    plt.imshow(A_gnd[0, daslice, :, :, 0], cmap='gray', vmin=0.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.title("Sobel Horizontal")
    plt.imshow(grad_op_x_A[0, daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.title("Sobel Vertical")
    plt.imshow(grad_op_y_A[0, daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(2, 4, 5)
    plt.title(B_name)
    plt.imshow(B[0, daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.title("Lesion overlay")
    base = (B[0, daslice, :, :, 0]+1)*0.5
    overlay = np.array([base+(A_gnd[0, daslice, :, :, 0]*0.3), base, base])
    overlay = np.moveaxis(overlay, 0, -1)
    plt.imshow(overlay)
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.title("Sobel Horizontal")
    plt.imshow(grad_op_x_B[0, daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.title("Sobel Vertical")
    plt.imshow(grad_op_y_B[0, daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
    plt.axis('off')

    plt.savefig(folder_prepend + 'sobel_' + A_name + "_" + num + '.png', dpi=200)
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

    # inverse
    """
    kernel_in_x_inv = np.array([ 
        [ [ [1] ],[ [0] ],[ [-1] ] ],
        [ [ [2] ],[ [0] ],[ [-2] ] ],
        [ [ [1] ],[ [0] ],[ [-1] ] ],
        ])
    kernel_x_inv = tf.constant(kernel_in_x_inv, dtype=tf.float32)

    kernel_in_y_inv = np.array([ 
        [ [ [-1] ],[ [-2] ],[ [-1] ] ],
        [ [ [0] ],[ [0] ],[ [0] ] ],
        [ [ [1] ],[ [2] ],[ [1] ] ],
        ])
    kernel_y_inv = tf.constant(kernel_in_y_inv, dtype=tf.float32)

    grad_op_x_B_inv = tf.nn.conv3d(grad_op_B, kernel_x_inv, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_op_y_B_inv = tf.nn.conv3d(grad_op_B, kernel_y_inv, strides=[1, 1, 1, 1, 1], padding='SAME')

    grad_op_x_B = (grad_op_x_B * A_gnd_inv) + (grad_op_x_B_inv * A_gnd)
    grad_op_y_B = (grad_op_y_B * A_gnd_inv) + (grad_op_y_B_inv * A_gnd)

    """

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
    
    return tf.math.reduce_sum(A_norm*B_norm)/(AB_root+0.0000000000000001)#((A_root*B_root)+0.00000000001)



learning_rate_disc = 2e-4 # last best: 2e-6  def: 2e-4  last: 5e-6
learning_rate_gen = 2e-4 # last best: 5e-4  last: 1e-4

generator_g_optimizer = tf.keras.optimizers.Adam(learning_rate_gen, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(learning_rate_gen, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(learning_rate_disc, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(learning_rate_disc, beta_1=0.5)

"""

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

"""


def generate_images(model, model_b, test_input, dagnd, label, datitle, dastruct):
    back = back_extract(test_input)
    prediction = model(test_input)
    prediction = ((prediction + 1.0) * back) - 1.0
    reconstructed = model_b(prediction)
    reconstructed = ((reconstructed + 1.0) * back) - 1.0

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = [datitle, dastruct]

    for i in range(2):
        plt.subplot(3, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i][daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
        plt.axis('off')

    plt.subplot(3, 2, 3)
    plt.title("gnd")
    plt.imshow(dagnd[0, daslice, :, :, 0], cmap='gray', vmin=0.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(3, 2, 4)
    plt.title("gnd_over")
    base = (prediction[0, daslice, :, :, 0]+1)*0.5
    overlay = np.array([base+(dagnd[0, daslice, :, :, 0]*0.3), base, base])
    plt.imshow(np.moveaxis(overlay, 0, -1))
    plt.axis('off')

    plt.subplot(3, 2, 5)
    plt.title("reconstructed")
    plt.imshow(reconstructed[0, daslice, :, :, 0], cmap='gray', vmin=-1.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(3, 2, 6)
    plt.title("difference")
    diff = np.absolute((display_list[0][daslice, :, :, 0]+1) - (display_list[1][daslice, :, :, 0]+1))
    damax = np.nanmax(diff)
    plt.imshow(diff/damax, cmap='gray', vmin=0, vmax=1.0)
    plt.axis('off')

    #plt.show()
    plt.savefig(folder_prepend + 'test/cyclegan_trained_' + label + '.png')
    plt.close()


def back_extract(in_img):
    binary = np.where(in_img[0,:,:,:,0]>-0.9, 1.0, 0.0) 
    filled = flood_fill(binary, (5, 5, 5), 100.0)
    back = np.where(filled>=100.0, 0.0, 1.0)
    return back[np.newaxis, ..., np.newaxis].astype('float32')



def compute_stats(my_folder_prepend, vanilla=True):

    folder_prepend = my_folder_prepend

    if vanilla:

        generator_g = pix2pix_van.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm', )
        #print(generator_g.summary())
        generator_g.load_weights(folder_prepend+"generator_mr_to_ct.h5")

        generator_f = pix2pix_van.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
        #print(generator_f.summary())
        generator_f.load_weights(folder_prepend+"generator_ct_to_mr.h5")

        discriminator_x = pix2pix_van.discriminator(norm_type='instancenorm', target=False)
        #print(generator_g.summary())
        discriminator_x.load_weights(folder_prepend+"discriminator_mr_to_ct.h5")

        discriminator_y = pix2pix_van.discriminator(norm_type='instancenorm', target=False)
        #print(generator_f.summary())
        discriminator_y.load_weights(folder_prepend+"discriminator_ct_to_mr.h5")

    else:

        generator_g, mid_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
        #print(generator_g.summary())
        generator_g.load_weights(folder_prepend+"generator_mr_to_ct.h5")

        generator_f, mid_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
        #print(generator_f.summary())
        generator_f.load_weights(folder_prepend+"generator_ct_to_mr.h5")

        discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
        #print(generator_g.summary())
        discriminator_x.load_weights(folder_prepend+"discriminator_mr_to_ct.h5")

        discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
        #print(generator_f.summary())
        discriminator_y.load_weights(folder_prepend+"discriminator_ct_to_mr.h5")

    accum_g = []
    struct_g = []
    gen_g = []
    disc_g = []
    accum_f = []
    struct_f = []
    gen_f = []
    disc_f = []

    if len(test_images_mr) > len(test_images_ct):
        lim_data = len(test_images_ct)
    else:
        lim_data = len(test_images_mr)


    for k in range(lim_data):

        real_x = test_images_mr[k]
        gnd_x = test_gnd_images_mr[k]
        real_y = test_images_ct[k]
        gnd_y = test_gnd_images_ct[k]

        fake_y = generator_g(real_x, training=False)
        cycled_x = generator_f(fake_y, training=False)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x)

        structural_loss_g = (GC(real_x, fake_y, gnd_x))

        disc_real_y = discriminator_y(tf.concat([real_y] if vanilla else [real_y, gnd_y], axis=4), training=False) #gnd_y
        disc_fake_y = discriminator_y(tf.concat([fake_y] if vanilla else [fake_y, gnd_x], axis=4), training=False) #gnd_x

        disc_loss = discriminator_loss(disc_real_y, disc_fake_y)
        gen_loss = generator_loss(disc_fake_y)

        accum_g.append(total_cycle_loss)
        struct_g.append(structural_loss_g)
        gen_g.append(gen_loss)
        disc_g.append(disc_loss)

        fake_x = generator_f(real_y, training=False)
        cycled_y = generator_g(fake_x, training=False)

        total_cycle_loss = calc_cycle_loss(real_y, cycled_y)

        structural_loss_f = (GC(real_y, fake_x, gnd_y))

        disc_real_x = discriminator_x(tf.concat([real_x] if vanilla else [real_x, gnd_x], axis=4), training=False) #x
        disc_fake_x = discriminator_x(tf.concat([fake_x] if vanilla else [fake_x, gnd_y], axis=4), training=False) # y
        disc_loss = discriminator_loss(disc_real_x, disc_fake_x)
        gen_loss = generator_loss(disc_fake_x)

        accum_f.append(total_cycle_loss)
        struct_f.append(structural_loss_f)
        gen_f.append(gen_loss)
        disc_f.append(disc_loss)


    struct_g = np.array(struct_g)
    struct_f = np.array(struct_f)

    return np.array([struct_g, struct_f])



model_source = ["./eval_4_all/vanilla/", "./eval_4_all/no_attention/", "./eval_4_all/no_struct/", "./eval_4_all/"]
model_name = ["CycleGAN", "CycleGAN\n+\nGC Loss", "CycleGAN\n+\nAttention", "Proposed\n(all combined)"]
model_vanilla = [True, True, False, False]

df_a = []
df_b = []
df_c = []

dadata = []

for i in range(len(model_source)):
    struct_all = compute_stats(model_source[i], model_vanilla[i])
    dadata.append(np.concatenate((struct_all[0], struct_all[1])))
    #dadata.append(struct_all[1])
    #print(struct_all[0])
    print(i)
    print(np.mean(struct_all[0]))
    #print(struct_all[1])
    print(np.mean(struct_all[1]))

stat01 = stats.ttest_rel(np.array(dadata[0]), np.array(dadata[1]))
cohen01 = cohens_d(np.array(dadata[0]), np.array(dadata[1]))
stat02 = stats.ttest_rel(np.array(dadata[0]), np.array(dadata[2]))
cohen02 = cohens_d(np.array(dadata[0]), np.array(dadata[2]))
stat03 = stats.ttest_rel(np.array(dadata[0]), np.array(dadata[3]))
cohen03 = cohens_d(np.array(dadata[0]), np.array(dadata[3]))
stat12 = stats.ttest_rel(np.array(dadata[1]), np.array(dadata[2]))
cohen12 = cohens_d(np.array(dadata[1]), np.array(dadata[2]))
stat13 = stats.ttest_rel(np.array(dadata[1]), np.array(dadata[3]))
cohen13 = cohens_d(np.array(dadata[1]), np.array(dadata[3]))
stat23 = stats.ttest_rel(np.array(dadata[2]), np.array(dadata[3]))
cohen23 = cohens_d(np.array(dadata[2]), np.array(dadata[3]))
#print(stats.ttest_rel(np.array(dadata[4]), np.array(dadata[6])))
#print(stats.ttest_rel(np.array(dadata[5]), np.array(dadata[7])))

print("01 - p:"+str(stat01.pvalue)+" t:"+str(stat01.statistic)+" d:"+str(cohen01))
print("02 - p:"+str(stat02.pvalue)+" t:"+str(stat02.statistic)+" d:"+str(cohen02))
print("03 - p:"+str(stat03.pvalue)+" t:"+str(stat03.statistic)+" d:"+str(cohen03))
print("12 - p:"+str(stat12.pvalue)+" t:"+str(stat12.statistic)+" d:"+str(cohen12))
print("13 - p:"+str(stat13.pvalue)+" t:"+str(stat13.statistic)+" d:"+str(cohen13))
print("23 - p:"+str(stat23.pvalue)+" t:"+str(stat23.statistic)+" d:"+str(cohen23))


for i in range(len(model_source)):

    # --------------------------------------
    # ----- Plot seaborn violin dists ------ 
    # --------------------------------------

    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
    sns.set(style="whitegrid")

    struct_all = compute_stats(model_source[i], model_vanilla[i])

    # struct g and f into pandas dataframe
    struct_g_label = np.full(struct_all[0].shape, "FLAIR to NCCT")
    struct_f_label = np.full(struct_all[1].shape, "NCCT to FLAIR")

    struct_g_model = np.full(struct_all[0].shape, model_name[i])
    struct_f_model = np.full(struct_all[1].shape, model_name[i])

    df_a.append(list(np.append(struct_all[0], struct_all[1]).astype('float32')))
    df_b.append(list(np.append(struct_g_label, struct_f_label)))
    df_c.append(list(np.append(struct_g_model, struct_f_model)))


df_a = np.concatenate((np.array(df_a[0]), np.array(df_a[1]), np.array(df_a[2]), np.array(df_a[3])))
df_b = np.concatenate((np.array(df_b[0]), np.array(df_b[1]), np.array(df_b[2]), np.array(df_b[3])))
df_c = np.concatenate((np.array(df_c[0]), np.array(df_c[1]), np.array(df_c[2]), np.array(df_c[3])))


df = pd.DataFrame(np.array([df_a, df_b, df_c]).T,
                            columns=["correlation", "label", "model"]
                )

new_dtypes = {"correlation": np.float32}
df = df.astype(new_dtypes)

print(df.shape)

# Grouped boxplot
box_plot = sns.boxplot(x="model", y="correlation", showmeans=True, meanprops={"marker":"x",
                       "markerfacecolor":"black", 
                       "markeredgecolor":"black",
                      "markersize":"6"}, data=df, palette="Pastel1")
medians = df.groupby(['model'])['correlation'].median()
vertical_offset = df['correlation'].median() * 0.05 # offset from median for display

#sns.stripplot(x="model", y="correlation", data=df, size=4, color=".3", linewidth=0)
#sns.violinplot(x="model", y="correlation", hue="label", data=df, palette="Pastel1") 
#box_plot = sns.lineplot(x="model", y="correlation", data=df, color="black", ci=None, legend=False, ax=box_plot)

"""
for item in df.groupby('model'):
    print(item)
    #item[1] is a grouped data frame
    for x,y in item[['x','y','mark_value']].values:
        ax.text(x,y,f'{m:.2f}',color=color)

"""

def p_asterix(p):

    if p > 0.05:
        return "ns"
    elif p > 0.01:
        return "*"
    elif p > 0.001:
        return "**"
    else:
        return "***"


# statistical annotation 1
x1, x2 = 0, 1
y, h, col = df['correlation'].max() + 0.005, 0.022, 'k'
plt.plot([x1, x2], [y+h, y+h], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h+0.001, f"{p_asterix(stat01.pvalue)} d={np.abs(cohen01):.2f}", 
    ha='center', va='bottom', color=col, fontsize=10)
#plt.text((x1+x2)*.5, y+h+0.009+0.001, f"p={stat1.pvalue:.2f}", ha='center', va='bottom', color=col, fontsize=10)
#plt.text((x1+x2)*.5, y+h+0.018+0.001, f"t={stat1.statistic:.2f}", ha='center', va='bottom', color=col, fontsize=10)

x1, x2 = 0, 2
y, h, col = df['correlation'].max() + 0.005, 0.011, 'k'
plt.plot([x1, x2], [y+h, y+h], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h+0.001, f"{p_asterix(stat02.pvalue)} d={np.abs(cohen02):.2f}", 
    ha='center', va='bottom', color=col, fontsize=10)

x1, x2 = 0, 3
y, h, col = df['correlation'].max() + 0.005, 0.000, 'k'
plt.plot([x1, x2], [y+h, y+h], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h+0.001, f"{p_asterix(stat03.pvalue)} d={np.abs(cohen03):.2f}", 
    ha='center', va='bottom', color=col, fontsize=10)

# statistical annotation 2
x1, x2 = 1, 2
y, h, col = df['correlation'].max() + 0.005, 0.044, 'k'
plt.plot([x1, x2], [y+h, y+h], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h+0.001, f"{p_asterix(stat12.pvalue)} d={np.abs(cohen12):.2f}", 
    ha='center', va='bottom', color=col, fontsize=10)
#plt.text((x1+x2)*.5, y+h+0.009+0.001, f"p={stat2.pvalue:.2f}", ha='center', va='bottom', color=col, fontsize=10)
#plt.text((x1+x2)*.5, y+h+0.018+0.001, f"t={stat2.statistic:.2f}", ha='center', va='bottom', color=col, fontsize=10)

# statistical annotation 3
x1, x2 = 1, 3
y, h, col = df['correlation'].max() + 0.005, 0.033, 'k'
plt.plot([x1, x2], [y+h, y+h], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h+0.001, f"{p_asterix(stat13.pvalue)} d={np.abs(cohen13):.2f}", 
    ha='center', va='bottom', color=col, fontsize=10)

x1, x2 = 2, 3
y, h, col = df['correlation'].max() + 0.005, 0.022, 'k'
plt.plot([x1, x2], [y+h, y+h], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h+0.001, f"{p_asterix(stat23.pvalue)} d={np.abs(cohen23):.2f}", 
    ha='center', va='bottom', color=col, fontsize=10)

plt.ylim(None, 1.04)
box_plot.set_xlabel(None)
box_plot.set_ylabel("Gradient correlation")

plt.savefig("test_plot_gradients.svg", bbox_inches='tight')
plt.close()
print("Violin plot complete.")

# -----------------------

struct_mean_g = struct_g.mean()
struct_mean_f = struct_f.mean()
struct_stdev_g = struct_g.std()
struct_stdev_f = struct_f.std()
struct_mean = ( struct_mean_g + struct_mean_f )/ 2
struct_stdev = ( struct_stdev_g + struct_stdev_f )/ 2

# -----------

accum_g = np.array(accum_g)
accum_f = np.array(accum_f)

accum_mean_g = accum_g.mean()
accum_mean_f = accum_f.mean()
accum_stdev_g = accum_g.std()
accum_stdev_f = accum_f.std()
accum_mean = ( accum_mean_g + accum_mean_f )/ 2
accum_stdev = ( accum_stdev_g + accum_stdev_f )/ 2

accum_g_ordered_idxs = np.sort(accum_g)
print(accum_g_ordered_idxs)
accum_g_ordered_idxs = range(len(accum_g))#np.argsort(accum_g)

accum_f_ordered_idxs = np.sort(accum_f)
print(accum_f_ordered_idxs)
accum_f_ordered_idxs = range(len(accum_f))#np.argsort(accum_f)

# --------------

gen_g = np.array(gen_g)
gen_f = np.array(gen_f)

gen_mean_g = gen_g.mean()
gen_mean_f = gen_f.mean()
gen_stdev_g = gen_g.std()
gen_stdev_f = gen_f.std()
gen_mean = ( gen_mean_g + gen_mean_f )/ 2
gen_stdev = ( gen_stdev_g + gen_stdev_f )/ 2

# -----------

disc_g = np.array(gen_g)
disc_f = np.array(disc_f)

disc_mean_g = disc_g.mean()
disc_mean_f = disc_f.mean()
disc_stdev_g = disc_g.std()
disc_stdev_f = disc_f.std()
disc_mean = ( disc_mean_g + disc_mean_f )/ 2
disc_stdev = ( disc_stdev_g + disc_stdev_f )/ 2

# -----------

# Run the trained model on the test dataset
print(len(accum_g))
print(len(accum_f))

#x_interv = math.floor((len(accum_x)-1)/19)
#y_interv = math.floor((len(accum_y)-1)/19)

g_interv = 1
f_interv = 1


for inp in range(len(accum_g)):
    print(str(inp*g_interv))
    cycle_loss = str(round(accum_g[accum_g_ordered_idxs[inp*g_interv]], 2))
    struct_loss = str(round(struct_g[accum_g_ordered_idxs[inp*g_interv]], 2))
    print(cycle_loss)
    generate_images(generator_g, generator_f, test_images_mr[accum_g_ordered_idxs[inp*g_interv]], test_gnd_images_mr[accum_g_ordered_idxs[inp*g_interv]], "test_mr_"+str(inp*g_interv), cycle_loss, struct_loss)



for inp in range(len(accum_f)):
    print(str(inp*f_interv))
    cycle_loss = str(round(accum_f[accum_f_ordered_idxs[inp*f_interv]], 2))
    struct_loss = str(round(struct_f[accum_f_ordered_idxs[inp*f_interv]], 2))
    print(cycle_loss)
    generate_images(generator_f, generator_g, test_images_ct[accum_f_ordered_idxs[inp*f_interv]], test_gnd_images_ct[accum_f_ordered_idxs[inp*f_interv]], "test_ct_"+str(inp*f_interv), cycle_loss, struct_loss)
"""


inp = 30
print(str(inp*g_interv))
cycle_loss = str(round(accum_g[accum_g_ordered_idxs[inp*g_interv]], 2))
struct_loss = str(round(struct_g[accum_g_ordered_idxs[inp*g_interv]], 2))
print(cycle_loss)
generate_images(generator_g, generator_f, test_images_mr[accum_g_ordered_idxs[inp*g_interv]], test_gnd_images_mr[accum_g_ordered_idxs[inp*g_interv]], "test_mr_"+str(inp*g_interv), cycle_loss, struct_loss)


inp = 30
print(str(inp*f_interv))
cycle_loss = str(round(accum_f[accum_f_ordered_idxs[inp*f_interv]], 2))
struct_loss = str(round(struct_f[accum_f_ordered_idxs[inp*f_interv]], 2))
print(cycle_loss)
generate_images(generator_f, generator_g, test_images_ct[accum_f_ordered_idxs[inp*f_interv]], test_gnd_images_ct[accum_f_ordered_idxs[inp*f_interv]], "test_ct_"+str(inp*f_interv), cycle_loss, struct_loss)

"""

print("struct_mean: " + str(struct_mean))
print("struct_stdev: " + str(struct_stdev))
print("struct_mean_g: " + str(struct_mean_g))
print("struct_stdev_g: " + str(struct_stdev_g))
print("struct_mean_f: " + str(struct_mean_f))
print("struct_stdev_f: " + str(struct_stdev_f))

print("---")

print("cycle_mean: " + str(accum_mean))
print("cycle_stdev: " + str(accum_stdev))
print("cycle_mean_g: " + str(accum_mean_g))
print("cycle_stdev_g: " + str(accum_stdev_g))
print("cycle_mean_f: " + str(accum_mean_f))
print("cycle_stdev_f: " + str(accum_stdev_f))

print("---")

print("gen_mean: " + str(gen_mean))
print("gen_stdev: " + str(gen_stdev))
print("gen_mean_g: " + str(gen_mean_g))
print("gen_stdev_g: " + str(gen_stdev_g))
print("gen_mean_f: " + str(gen_mean_f))
print("gen_stdev_f: " + str(gen_stdev_f))

print("---")

print("disc_mean: " + str(disc_mean))
print("disc_stdev: " + str(disc_stdev))
print("disc_mean_g: " + str(disc_mean_g))
print("disc_stdev_g: " + str(disc_stdev_g))
print("disc_mean_f: " + str(disc_mean_f))
print("disc_stdev_f: " + str(disc_stdev_f))

