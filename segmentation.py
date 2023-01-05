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
seg_unet.summary()

loss = []
val_loss = []


for epoch in range(EPOCHS):

    start = time.time()
    daloss = []
    
    src_idxs = np.array(range(images_mr_names.shape[0]))
    #src_idxs = [1]*90
    np.random.shuffle(src_idxs)
    src_images_shuffled = images_mr_names[src_idxs]
    src_lesions_shuffled = lesions_mr_names[src_idxs]

    batches = int(np.floor(len(src_images_shuffled)/BATCH_SIZE))
    last_batch = len(src_images_shuffled)-(batches*BATCH_SIZE)


    for pr in range(batches):

        offset_batch = pr*BATCH_SIZE

        image = []
        lesion = []

        for prr in range(BATCH_SIZE):
            image.append(getimg(src_images_shuffled[offset_batch+prr]))
            lesion.append(tf.keras.utils.to_categorical(getimg(src_lesions_shuffled[offset_batch+prr]), num_classes=2, dtype='float32'))

            
        image = np.concatenate(tuple(image), axis=0, dtype="float32")
        lesion = np.concatenate(tuple(lesion), axis=0, dtype="float32")

        dalossy = train_step_seg(image, lesion) 
        daloss.append(dalossy)
           
        print("dataset "+str(pr+1)+"/"+str(round(len(src_images_shuffled)/BATCH_SIZE))+" loss:"+str(np.array(daloss).mean()))



    generate_images(seg_unet, getimg(src_images_shuffled[10]), getimg(src_lesions_shuffled[10]), "tra_pred_e"+str(epoch))
    generate_images(seg_unet, val_image, val_lesion, "val_pred_e"+str(epoch))

    print(np.nanmax(seg_unet(val_image).numpy()))
    print(np.nanmin(seg_unet(val_image).numpy()))

    #print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    daloss = np.array(daloss)
    daloss = daloss.mean()

    daloss_val = evaluate_loss()
    print('Prediction loss: ' + str(daloss) + ' - val loss: ' + str(daloss_val))
    
    loss.append(daloss)
    val_loss.append(daloss_val)

    if (epoch+1)%15 == 0:

        epochss = range(epoch+1)

        plt.figure()
        plt.plot(epochss, loss, 'c', label='Prediction loss', linewidth=0.5)
        plt.plot(epochss, val_loss, 'm', label='Validation loss', linewidth=0.5)
        plt.title('loss', fontsize=10)
        plt.legend()


if direction == "ct_to_mr":
    seg_unet.save_weights(folder_prepend + 'seg_mr.h5')
    seg_unet.save(folder_prepend + 'seg_mr')
else:
    seg_unet.save_weights(folder_prepend + 'seg_ct.h5')
    seg_unet.save(folder_prepend + 'seg_ct')


mr_to_ct, mid_g = pix2pix.unet_generator(1, norm_type='instancenorm')
ct_to_mr, mid_f = pix2pix.unet_generator(1, norm_type='instancenorm')

ct_to_mr.summary()
    
if direction == "ct_to_mr":
    mr_to_ct.load_weights(folder_prepend+"generator_mr_to_ct.h5")
    ct_to_mr.load_weights(folder_prepend+"generator_ct_to_mr.h5")
else:
    mr_to_ct.load_weights(folder_prepend+"generator_ct_to_mr.h5")
    ct_to_mr.load_weights(folder_prepend+"generator_mr_to_ct.h5")    



accum_y = []
accum_yy = []

for kk in range(test_images_mr_names.shape[0]):

    image_x = getimg(test_images_mr_names[kk])
    gt = tf.keras.utils.to_categorical(getimg(test_lesions_mr_names[kk]), num_classes=2, dtype='float32')

    y = seg_unet(image_x, training=False)
    seg_loss = 1-alt_dice(y, gt)
    seg_loss_b = alt_dicee(y.numpy(), gt)

    accum_y.append(seg_loss)
    accum_yy.append(seg_loss_b)

accum_y = np.array(accum_y)
accum_y = accum_y.mean()

print("Test DSC: " + str(accum_y))



# ----------------------------------------------

accum_y = []
accum_yy = []

for kk in range(test_images_ct_names.shape[0]):

    image_x = getimg(test_images_ct_names[kk])
    gt = tf.keras.utils.to_categorical(getimg(test_lesions_ct_names[kk]), num_classes=2, dtype='float32')

    image_z = ct_to_mr(image_x)

    y = seg_unet(image_z, training=False)
    seg_loss = 1-alt_dice(y, gt)
    seg_loss_b = alt_dicee(y.numpy(), gt)

    accum_y.append(seg_loss)
    accum_yy.append(seg_loss_b)

accum_y = np.array(accum_y)
accum_y = accum_y.mean()

print("Test ("+str(direction)+") DSC: " + str(accum_y))


