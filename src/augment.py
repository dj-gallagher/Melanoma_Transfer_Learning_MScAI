"""
Created by: Daniel Gallagher
Date: 23 March 2022

source: https://stackoverflow.com/questions/71058783/how-to-add-augmented-images-to-original-dataset-with-tensorflow

source: https://stackoverflow.com/questions/51136559/mixing-augmented-and-original-samples-using-tf-data-dataset-map
"""

import matplotlib.pyplot as plt
from numpy import True_
import tensorflow as tf
from math import ceil

'''test_ds, test_size, test_labels = read_test_csv_to_dataset()

batch_size = 32
test_steps = ceil(test_size / batch_size)

test_size = 2

test = rescale_and_resize(ds=test_ds,
                            ds_size=test_size,
                            batch_size=1,
                            training_set=True)

test = test.unbatch()'''

def augment_image(image, label, 
                  brightness=False,
                  contrast=False,
                  crop=False,
                  horizontal_flip=False,
                  vertical_flip=False,
                  rotate=False,
                  num_rotations=1):
    """
    Image augmentation mapper function.

    This function maps the training set with various image augmentations to generalise the training set.

    :param file_path: The path to the image file to augment.
    :param label: The label associated with the image.
    :param conf: The flow configuration.
    :return: The augmented image and its label.
    """
    # Get the image, resized and rescaled.
    #image, label = rescale_and_resize_image(file_path, label, )

    seed = (1,2)
    # Brightness Augmentation
    # =======================================================================================================================
    if brightness:
        image = tf.image.stateless_random_brightness(image,
                                                    max_delta=0.2,
                                                    seed=seed)
    # Contrast Augmentation
    # =======================================================================================================================
    if contrast:
        image = tf.image.stateless_random_contrast(image,
                                                  lower=0.2,
                                                  upper=0.5,
                                                  seed=seed)
    # Crop Augmentation
    # =======================================================================================================================
    if crop:
        image = tf.image.stateless_random_crop(image,
                                              size=[224,
                                                    224,
                                                    3],
                                              seed=seed)
    # Horizontal Flip
    # =======================================================================================================================
    if horizontal_flip:
        image = tf.image.flip_left_right(image)
    # Vertical Flip
    # =======================================================================================================================
    if vertical_flip:
        image = tf.image.flip_up_down(image)
        
    # Rotation
    # =======================================================================================================================
    if rotate:
        image = tf.image.rot90(image, k=num_rotations)
    '''
    # Hue Augmentation
    # =======================================================================================================================
    if hue:
        tf.image.stateless_random_hue(image,
                                      max_delta=conf.ppro.aug.hue.delta,
                                      seed=conf.seed2)

    # JPEG Quality Augmentation
    # =======================================================================================================================
    if jpeg_quality:
        tf.image.stateless_random_jpeg_quality(image,
                                               min_jpeg_quality=conf.ppro.aug.jpeg_quality.min_jpeg_quality,
                                               max_jpeg_quality=conf.ppro.aug.jpeg_quality.max_jpeg_quality,
                                               seed=conf.seed2)
    # Saturation Augmentation
    # =======================================================================================================================
    if saturation:
        tf.image.stateless_random_saturation(image,
                                             lower=conf.ppro.aug.saturation.lower,
                                             upper=conf.ppro.aug.saturation.upper,
                                             seed=conf.seed2)
    '''
    return image, label

def augment_dataset(ds, ds_size, augment):
    """
    Must be applied to an unbatched, no repeating tf dataset.
    """

    if augment=="Mahbod":
        ds_size = ds_size*8
        
        # rotation by 90
        aug_1 = ds.map( lambda image, label: augment_image(image, label, rotate=True, num_rotations=1) )
        
        # rotation by 180
        aug_2 = ds.map( lambda image, label: augment_image(image, label, rotate=True, num_rotations=2) )
        
        # rotation by 270
        aug_3 = ds.map( lambda image, label: augment_image(image, label, rotate=True, num_rotations=3) )
        
        # flip horizontal
        aug_4 = ds.map( lambda image, label: augment_image(image, label, horizontal_flip=True) )
        
        # flip rot 90
        #aug_5 = (ds.map( lambda image, label: augment_image(image, label, rotate=True, num_rotations=1) )
        #        .map( lambda image, label: augment_image(image, label, horizontal_flip=True) ) )
        aug_5 = ds.map( lambda image, label: augment_image(image, label, rotate=True, num_rotations=1, horizontal_flip=True) )
        
        # flip rot 180
        #aug_6 = (ds.map( lambda image, label: augment_image(image, label, rotate=True, num_rotations=2) )
        #        .map( lambda image, label: augment_image(image, label, horizontal_flip=True) ) )
        aug_6 = ds.map( lambda image, label: augment_image(image, label, rotate=True, num_rotations=2, horizontal_flip=True) )

        # flip rot 270
        #aug_7 = (ds.map( lambda image, label: augment_image(image, label, rotate=True, num_rotations=3) )
        #        .map( lambda image, label: augment_image(image, label, horizontal_flip=True) ) )
        aug_7 = ds.map( lambda image, label: augment_image(image, label, rotate=True, num_rotations=3, horizontal_flip=True) )

        ds = (ds.concatenate(aug_1)
                .concatenate(aug_2)
                .concatenate(aug_3)
                .concatenate(aug_4)
                .concatenate(aug_5)
                .concatenate(aug_6)
                .concatenate(aug_7))
        
    elif augment=="Hosseinzadeh":
        
        ds_size = ds_size*3
        
        # horizontal flip
        aug_1 = ds.map( lambda image, label: augment_image(image, label, horizontal_flip=True) )
        
        # vertical flip
        aug_2 = ds.map( lambda image, label: augment_image(image, label, vertical_flip=True) )
        
        # random brightness
        #aug_3 = ds.map( lambda image, label: augment_image(image, label, brightness=True) )
        
        # random contrast
        #aug_4 = ds.map( lambda image, label: augment_image(image, label, contrast=True) )
        
        ds = (ds.concatenate(aug_1)
                .concatenate(aug_2))


    #ds = ds.shuffle(buffer_size=ds_size)        
    
    # Return all datesets combined and new dataset size
    return ds, ds_size