import tensorflow as tf
import tensorflow_datasets as tfds
import patientkeyinfo as pki
import SimpleITK as sitk
import pix2pix3D_backup as pix2pix
import numpy as np
import scipy
import sys
from scipy import stats
from scipy import ndimage
from skimage.segmentation import flood, flood_fill
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import metrics
import psutil
import gc
import elastic_augmentation as ea
import glob

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output


# load scans NCCT and FLAIR

np.random.seed(0)

LAMBDA = 1
EPOCHS = 100
learning_rate = 2e-4
BATCH_SIZE = 3
OUTPUT_CHANNELS = 2


if(len(sys.argv)>1):
    direction = str(sys.argv[1])
else:
    direction = "ct_to_mr"


folder_prepend = "./eval_4_all/"
folder_prepend_model = "./eval_4_all/no_attention/"


def getimg(name):
    aa = np.load(name).astype("float32")
    return aa


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


def extract_names(folder):
    names = glob.glob(folder, recursive=False)
    names = sorted(names)

    assert len(names) > 0, "Could not find any training data"
    print("Number of samples: ", len(names))

    return np.array(names)


if direction == "ct_to_mr":

    images_mr_names = extract_names("./pre_processed/npy/mr/images/images/train/*")
    images_ct_names = extract_names("./pre_processed/npy/ct/images/images/train/*")
    images_mr_names_flipped = extract_names("./pre_processed/npy/mr/flipped/images/train/*")
    images_ct_names_flipped = extract_names("./pre_processed/npy/ct/flipped/images/train/*")
    aug_images_mr_names = extract_names("./pre_processed/npy/mr/augmentations/images/train/*")
    aug_images_ct_names = extract_names("./pre_processed/npy/ct/augmentations/images/train/*")
    aug_images_mr_names_flipped = extract_names("./pre_processed/npy/mr/augmentations_flipped/images/train/*")
    aug_images_ct_names_flipped = extract_names("./pre_processed/npy/ct/augmentations_flipped/images/train/*")
    val_images_mr_names = extract_names("./pre_processed/npy/mr/images/images/val/*")
    val_images_ct_names = extract_names("./pre_processed/npy/ct/images/images/val/*")
    test_images_mr_names = extract_names("./pre_processed/npy/mr/images/images/test/*")
    test_images_ct_names = extract_names("./pre_processed/npy/ct/images/images/test/*")

    lesions_mr_names = extract_names("./pre_processed/npy/mr/images/lesions/train/*")
    lesions_ct_names = extract_names("./pre_processed/npy/ct/images/lesions/train/*")
    lesions_mr_names_flipped = extract_names("./pre_processed/npy/mr/flipped/lesions/train/*")
    lesions_ct_names_flipped = extract_names("./pre_processed/npy/ct/flipped/lesions/train/*")
    aug_lesions_mr_names = extract_names("./pre_processed/npy/mr/augmentations/lesions/train/*")
    aug_lesions_ct_names = extract_names("./pre_processed/npy/ct/augmentations/lesions/train/*")
    aug_lesions_mr_names_flipped = extract_names("./pre_processed/npy/mr/augmentations_flipped/lesions/train/*")
    aug_lesions_ct_names_flipped = extract_names("./pre_processed/npy/ct/augmentations_flipped/lesions/train/*")
    val_lesions_mr_names = extract_names("./pre_processed/npy/mr/images/lesions/val/*")
    val_lesions_ct_names = extract_names("./pre_processed/npy/ct/images/lesions/val/*")
    test_lesions_mr_names = extract_names("./pre_processed/npy/mr/images/lesions/test/*")
    test_lesions_ct_names = extract_names("./pre_processed/npy/ct/images/lesions/test/*")

else:

    images_mr_names = extract_names("./pre_processed/npy/ct/images/images/train/*")
    images_ct_names = extract_names("./pre_processed/npy/mr/images/images/train/*")
    images_mr_names_flipped = extract_names("./pre_processed/npy/ct/flipped/images/train/*")
    images_ct_names_flipped = extract_names("./pre_processed/npy/mr/flipped/images/train/*")
    aug_images_mr_names = extract_names("./pre_processed/npy/ct/augmentations/images/train/*")
    aug_images_ct_names = extract_names("./pre_processed/npy/mr/augmentations/images/train/*")
    aug_images_mr_names_flipped = extract_names("./pre_processed/npy/ct/augmentations_flipped/images/train/*")
    aug_images_ct_names_flipped = extract_names("./pre_processed/npy/mr/augmentations_flipped/images/train/*")
    val_images_mr_names = extract_names("./pre_processed/npy/ct/images/images/val/*")
    val_images_ct_names = extract_names("./pre_processed/npy/mr/images/images/val/*")
    test_images_mr_names = extract_names("./pre_processed/npy/ct/images/images/test/*")
    test_images_ct_names = extract_names("./pre_processed/npy/mr/images/images/test/*")

    lesions_mr_names = extract_names("./pre_processed/npy/ct/images/lesions/train/*")
    lesions_ct_names = extract_names("./pre_processed/npy/mr/images/lesions/train/*")
    lesions_mr_names_flipped = extract_names("./pre_processed/npy/ct/flipped/lesions/train/*")
    lesions_ct_names_flipped = extract_names("./pre_processed/npy/mr/flipped/lesions/train/*")
    aug_lesions_mr_names = extract_names("./pre_processed/npy/ct/augmentations/lesions/train/*")
    aug_lesions_ct_names = extract_names("./pre_processed/npy/mr/augmentations/lesions/train/*")
    aug_lesions_mr_names_flipped = extract_names("./pre_processed/npy/ct/augmentations_flipped/lesions/train/*")
    aug_lesions_ct_names_flipped = extract_names("./pre_processed/npy/mr/augmentations_flipped/lesions/train/*")
    val_lesions_mr_names = extract_names("./pre_processed/npy/ct/images/lesions/val/*")
    val_lesions_ct_names = extract_names("./pre_processed/npy/mr/images/lesions/val/*")
    test_lesions_mr_names = extract_names("./pre_processed/npy/ct/images/lesions/test/*")
    test_lesions_ct_names = extract_names("./pre_processed/npy/mr/images/lesions/test/*")


val_image = getimg(val_images_mr_names[5])
val_lesion = getimg(val_lesions_mr_names[5])


process_m = psutil.Process(os.getpid())
print("A - RAM %.2f Gb" % (process_m.memory_info().rss/1000/1000/1000))


def alt_dicee(im1, im2):
    im1 = im1[0, :, :, :, 1]
    im2 = im2[0, :, :, :, 1]
    im1 = np.where(im1>0.5, 1.0, 0.0)
    im2 = np.where(im2>0.5, 1.0, 0.0)
    intersection = im1 * im2
    return (2.0 * np.sum(intersection)) / (np.sum(im1) + np.sum(im2) + 0.00000000001)


mydice = metrics.Dice(nb_labels=2, dice_type='soft')
alt_dice = mydice.loss


def unet(input_img):

    # encoder
    
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(input_img)
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1) 
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2) 
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3) 
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
    #conv4 = Dropout(0.4)(conv4)

    # decoder
    up1 = UpSampling3D((2, 2, 2))(conv4)
    merge1 = concatenate([up1, conv3], axis=4)
    merge1 = BatchNormalization()(merge1) 
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(merge1)
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv5)
    conv5 = Dropout(0.4)(conv5)
    
    up2 = UpSampling3D((2, 2, 2))(conv5)
    merge2 = concatenate([up2, conv2], axis=4)
    merge2 = BatchNormalization()(merge2) 
    conv6 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(merge2)
    conv6 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv6)
    conv6 = Dropout(0.4)(conv6)

    up3 = UpSampling3D((2, 2, 2))(conv6)
    merge3 = concatenate([up3, conv1], axis=4)
    merge3 = BatchNormalization()(merge3) 
    conv7 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(merge3)
    conv7 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv7)

    decoded = Conv3D(2, (3, 3, 3), activation='softmax', padding='same')(conv7)
    return decoded





def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[64, 160, 160, 1])
    return tf.keras.Model(inputs, unet(inputs))


def generate_images(model, test_input, ground_truth, label):

    to_tar = model(test_input)
        
    plt.tight_layout()
    plt.subplot(3, 3, 1)
    plt.title("source")
    plt.imshow(test_input[0, 16, :, :, -1], cmap='gray', vmin=-1.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.title("predicted lesion")
    plt.imshow(to_tar[0, 16, :, :, -1], cmap='gray', vmin=0.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.title("ground truth")
    plt.imshow(ground_truth[0, 16, :, :, -1], cmap='gray', vmin=0.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(test_input[0, 32, :, :, -1], cmap='gray', vmin=-1.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(to_tar[0, 32, :, :, -1], cmap='gray', vmin=0.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(ground_truth[0, 32, :, :, -1], cmap='gray', vmin=0.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.imshow(test_input[0, 48, :, :, -1], cmap='gray', vmin=-1.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.imshow(to_tar[0, 48, :, :, -1], cmap='gray', vmin=0.0, vmax=1.0)
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.imshow(ground_truth[0, 48, :, :, -1], cmap='gray', vmin=0.0, vmax=1.0)
    plt.axis('off')

    plt.savefig(folder_prepend + label + '.jpg', dpi=300)
    plt.close()


@tf.function
def train_step_seg(x, gt):


    with tf.GradientTape(persistent=True) as tape:

        y = seg_unet(x, training=True)
        seg_loss = LAMBDA * alt_dice(y, gt)

        seg_gradients = tape.gradient(seg_loss, seg_unet.trainable_variables)

        opt.apply_gradients(zip(seg_gradients, seg_unet.trainable_variables))


    return seg_loss


def evaluate_loss():

    accum_y = []

    for kk in range(val_images_mr_names.shape[0]):

        image_x = getimg(val_images_mr_names[kk])
        gt = tf.keras.utils.to_categorical(getimg(val_lesions_mr_names[kk]), num_classes=2, dtype='float32')
        
        y = seg_unet(image_x, training=False)
        seg_loss = LAMBDA * alt_dice(y, gt)

        accum_y.append(seg_loss)

    accum_y = np.array(accum_y)
    accum_y = accum_y.mean()

    return accum_y



opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)
seg_unet = unet_model(OUTPUT_CHANNELS)
seg_unet_B = unet_model(OUTPUT_CHANNELS)
seg_unet.summary()



if direction == "ct_to_mr":
    seg_unet.load_weights(folder_prepend + 'seg_mr.h5')
    seg_unet_B.load_weights(folder_prepend + 'seg_ct.h5')
    #seg_unet.save(folder_prepend + 'seg_mr')
else:
    seg_unet.load_weights(folder_prepend + 'seg_ct.h5')
    seg_unet_B.load_weights(folder_prepend + 'seg_mr.h5')
    #seg_unet.save(folder_prepend + 'seg_ct')




mr_to_ct, mid_g = pix2pix.unet_generator(1, norm_type='instancenorm')
ct_to_mr, mid_f = pix2pix.unet_generator(1, norm_type='instancenorm')

ct_to_mr.summary()
    
if direction == "ct_to_mr":
    mr_to_ct.load_weights(folder_prepend_model+"generator_mr_to_ct.h5")
    ct_to_mr.load_weights(folder_prepend_model+"generator_ct_to_mr.h5")
else:
    mr_to_ct.load_weights(folder_prepend_model+"generator_ct_to_mr.h5")
    ct_to_mr.load_weights(folder_prepend_model+"generator_mr_to_ct.h5")    


# absolute volume error
def AVE(im1, im2):
    im1 = im1[0, :, :, :, 1]
    im2 = im2[0, :, :, :, 1]
    im1 = np.where(im1>0.5, 1.0, 0.0)
    im2 = np.where(im2>0.5, 1.0, 0.0)
    vol1 = np.sum(im1)
    vol2 = np.sum(im2)
    return np.absolute(vol1 - vol2)


accum_y = []
accum_yy = []
volumes = []
MAVE = []
count = 0


paired_t_A = []
paired_t_B = []

for kk in range(test_images_mr_names.shape[0]):

    image_x = getimg(test_images_mr_names[kk])
    gt = tf.keras.utils.to_categorical(getimg(test_lesions_mr_names[kk]), num_classes=2, dtype='float32')

    y = seg_unet(image_x, training=False)
    seg_loss = 1-alt_dice(y, gt)
    seg_loss_b = alt_dicee(y.numpy(), gt)

    accum_y.append(seg_loss)
    accum_yy.append(seg_loss_b)

    volumes.append(np.sum(gt[0, :, :, :, 1]))
    MAVE.append(AVE(gt, y))
    count = count + 1


for kk in range(val_images_mr_names.shape[0]):

    image_x = getimg(val_images_mr_names[kk])
    gt = tf.keras.utils.to_categorical(getimg(val_lesions_mr_names[kk]), num_classes=2, dtype='float32')

    y = seg_unet(image_x, training=False)
    seg_loss = 1-alt_dice(y, gt)
    seg_loss_b = alt_dicee(y.numpy(), gt)

    accum_y.append(seg_loss)
    accum_yy.append(seg_loss_b)

    volumes.append(np.sum(gt[0, :, :, :, 1]))
    MAVE.append(AVE(gt, y))
    count = count + 1


paired_t_A = np.array(MAVE)

accum_y = np.array(accum_y)
accum_y_std = accum_y.std()
accum_y = accum_y.mean()

print("Test DSC: " + str(round(accum_y, 2)) + " std: " + str(round(accum_y_std, 2)))

accum_yy = np.array(accum_yy)
accum_yy_std = accum_yy.std()
accum_yy = accum_yy.mean()

print("Test hard DSC: " + str(round(accum_yy, 2)) + " std: " + str(round(accum_yy_std, 2)))

volumes = np.array(volumes)
plt.suptitle("Test set volume distribution")
plt.tight_layout()
plt.style.use('dark_background')
plt.hist(volumes, bins = 20, color = 'grey', edgecolor = 'white')
plt.savefig(folder_prepend_model + 'volumes_mr.jpg', dpi=300)
plt.close()
volumes_std = volumes.std()/1000
volumes = volumes.mean()/1000


print("Test volumes mr: " + str(round(volumes, 2)) + " std: " + str(round(volumes_std, 2)))

MAVE = np.array(MAVE)
MAVE_std = MAVE.std()/1000
MAVE = MAVE.mean()/1000

print("Test MAVE: " + str(round(MAVE, 2)) + " std: " + str(round(MAVE_std, 2)))

print(str(count))

# ----------------------------------------------

accum_y_B = []
accum_y_By = []
volumes_B = []
volumes_translated_B = []
MAVE_B = []
count_B = 0

for kk in range(test_images_ct_names.shape[0]):

    image_x = getimg(test_images_ct_names[kk])
    gt = tf.keras.utils.to_categorical(getimg(test_lesions_ct_names[kk]), num_classes=2, dtype='float32')

    image_z = ct_to_mr(image_x)

    y = seg_unet(image_z, training=False)
    seg_loss = 1-alt_dice(y, gt)
    seg_loss_b = alt_dicee(y.numpy(), gt)

    accum_y_B.append(seg_loss)
    accum_y_By.append(seg_loss_b)

    volumes_B.append(np.sum(gt[0, :, :, :, 1]))
    volumes_translated_B.append(np.sum(np.where(y[0, :, :, :, 1]>0.5, 1.0, 0.0)))
    MAVE_B.append(AVE(gt, y))
    count_B = count_B + 1



for kk in range(val_images_ct_names.shape[0]):

    image_x = getimg(val_images_ct_names[kk])
    gt = tf.keras.utils.to_categorical(getimg(val_lesions_ct_names[kk]), num_classes=2, dtype='float32')

    image_z = ct_to_mr(image_x)

    y = seg_unet(image_z, training=False)
    seg_loss = 1-alt_dice(y, gt)
    seg_loss_b = alt_dicee(y.numpy(), gt)

    accum_y_B.append(seg_loss)
    accum_y_By.append(seg_loss_b)

    volumes_B.append(np.sum(gt[0, :, :, :, 1]))
    volumes_translated_B.append(np.sum(np.where(y[0, :, :, :, 1]>0.5, 1.0, 0.0)))
    MAVE_B.append(AVE(gt, y))
    count_B = count_B + 1



print("\n"+direction)

accum_y_B = np.array(accum_y_B)
accum_y_B_std = accum_y_B.std()
accum_y_B = accum_y_B.mean()

print("Test DSC: " + str(round(accum_y_B, 2)) + " std: " + str(round(accum_y_B_std, 2)))

accum_y_By = np.array(accum_y_By)
accum_y_By_std = accum_y_By.std()
accum_y_By = accum_y_By.mean()

print("Test hard DSC: " + str(round(accum_y_By, 2)) + " std: " + str(round(accum_y_By_std, 2)))

volumes_B = np.array(volumes_B)
plt.suptitle("Test set volume distribution")
plt.tight_layout()
plt.style.use('dark_background')
plt.hist(volumes_B, bins = 20, color = 'grey', edgecolor = 'white')
plt.savefig(folder_prepend_model + 'volumes_ct.jpg', dpi=300)
plt.close()
volumes_B_std = volumes_B.std()/1000
volumes_B = volumes_B.mean()/1000


print("Test volumes_B_ct: " + str(round(volumes_B, 2)) + " std: " + str(round(volumes_B_std, 2)))


volumes_translated_B = np.array(volumes_translated_B)
plt.suptitle("Test set volume_translated distribution")
plt.tight_layout()
plt.style.use('dark_background')
plt.hist(volumes_translated_B, bins = 20, color = 'grey', edgecolor = 'white')
plt.savefig(folder_prepend_model + 'volumes_translated_B_ct_to_mr.jpg', dpi=300)
plt.close()
volumes_translated_B_std = volumes_translated_B.std()/1000
volumes_translated_B = volumes_translated_B.mean()/1000

print("Test volumes_translated_B_ct_to_mr: " + str(round(volumes_translated_B, 2)) + " std: " + str(round(volumes_translated_B_std, 2)))

MAVE_B = np.array(MAVE_B)
MAVE_B_std = MAVE_B.std()/1000
MAVE_B = MAVE_B.mean()/1000

print("Test MAVE_B: " + str(round(MAVE_B, 2)) + " std: " + str(round(MAVE_B_std, 2)))



print(str(count_B))


# ----------------------------------------------

accum_y_B = []
accum_y_By = []
volumes_B = []
volumes_translated_B = []
MAVE_B = []
count_B = 0

for kk in range(test_images_mr_names.shape[0]):

    image_x = getimg(test_images_mr_names[kk])
    gt = tf.keras.utils.to_categorical(getimg(test_lesions_mr_names[kk]), num_classes=2, dtype='float32')

    image_z = mr_to_ct(image_x)

    y = seg_unet_B(image_z, training=False)
    seg_loss = 1-alt_dice(y, gt)
    seg_loss_b = alt_dicee(y.numpy(), gt)

    accum_y_B.append(seg_loss)
    accum_y_By.append(seg_loss_b)

    volumes_B.append(np.sum(gt[0, :, :, :, 1]))
    volumes_translated_B.append(np.sum(np.where(y[0, :, :, :, 1]>0.5, 1.0, 0.0)))
    MAVE_B.append(AVE(gt, y))
    count_B = count_B + 1



for kk in range(val_images_mr_names.shape[0]):

    image_x = getimg(val_images_mr_names[kk])
    gt = tf.keras.utils.to_categorical(getimg(val_lesions_mr_names[kk]), num_classes=2, dtype='float32')

    image_z = mr_to_ct(image_x)

    y = seg_unet_B(image_z, training=False)
    seg_loss = 1-alt_dice(y, gt)
    seg_loss_b = alt_dicee(y.numpy(), gt)

    accum_y_B.append(seg_loss)
    accum_y_By.append(seg_loss_b)

    volumes_B.append(np.sum(gt[0, :, :, :, 1]))
    volumes_translated_B.append(np.sum(np.where(y[0, :, :, :, 1]>0.5, 1.0, 0.0)))
    MAVE_B.append(AVE(gt, y))
    count_B = count_B + 1



print("\nnot "+direction)
paired_t_B = np.array(MAVE_B)

accum_y_B = np.array(accum_y_B)
accum_y_B_std = accum_y_B.std()
accum_y_B = accum_y_B.mean()

print("Test DSC: " + str(round(accum_y_B, 2)) + " std: " + str(round(accum_y_B_std, 2)))

accum_y_By = np.array(accum_y_By)
accum_y_By_std = accum_y_By.std()
accum_y_By = accum_y_By.mean()

print("Test hard DSC: " + str(round(accum_y_By, 2)) + " std: " + str(round(accum_y_By_std, 2)))

volumes_translated_B = np.array(volumes_translated_B)
plt.suptitle("Test set volume_translated distribution")
plt.tight_layout()
plt.style.use('dark_background')
plt.hist(volumes_translated_B, bins = 20, color = 'grey', edgecolor = 'white')
plt.savefig(folder_prepend_model + 'volumes_translated_B_mr_to_ct.jpg', dpi=300)
plt.close()
volumes_translated_B_std = volumes_translated_B.std()/1000
volumes_translated_B = volumes_translated_B.mean()/1000

print("Test volumes_translated_B_mr_to_ct: " + str(round(volumes_translated_B, 2)) + " std: " + str(round(volumes_translated_B_std, 2)))

MAVE_B = np.array(MAVE_B)
MAVE_B_std = MAVE_B.std()/1000
MAVE_B = MAVE_B.mean()/1000

print("Test MAVE_B: " + str(round(MAVE_B, 2)) + " std: " + str(round(MAVE_B_std, 2)))



print(str(count_B))

print(scipy.stats.ttest_rel(paired_t_A, paired_t_B))