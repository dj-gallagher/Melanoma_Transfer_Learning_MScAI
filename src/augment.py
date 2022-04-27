"""
Created by: Daniel Gallagher
Date: 23 March 2022

source: https://stackoverflow.com/questions/71058783/how-to-add-augmented-images-to-original-dataset-with-tensorflow

source: https://stackoverflow.com/questions/51136559/mixing-augmented-and-original-samples-using-tf-data-dataset-map
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from math import ceil
import cv2


def augment_image(image, label, 
                  brightness=False,
                  contrast=False,
                  crop=False,
                  horizontal_flip=False,
                  vertical_flip=False,
                  rotate=False,
                  num_rotations=1,
                  grayworld=True):
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

    seed = 42
    # Brightness Augmentation
    # =======================================================================================================================
    if brightness:
        image = tf.image.random_brightness(image,
                                            max_delta=0.2,
                                            seed=seed)
    # Contrast Augmentation
    # =======================================================================================================================
    if contrast:
        image = tf.image.random_contrast(image,
                                        lower=0.2,
                                        upper=0.5,
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
        
    #if grayworld:
        #image = cv2.xphoto_GrayworldWB.getSaturationThreshold()
    
    return image, label

def augment_dataset(ds, ds_size, augment):
    """
    Must be applied to an unbatched, no repeating tf dataset.
    """

    if augment=="Mahbod":
        ds_size = ds_size*8
        
        # rotation by 90
        aug_1 = ds.map( lambda image, label: augment_image(image, label, rotate=True, num_rotations=1), tf.data.experimental.AUTOTUNE )
        
        # rotation by 180
        aug_2 = ds.map( lambda image, label: augment_image(image, label, rotate=True, num_rotations=2), tf.data.experimental.AUTOTUNE )
        
        # rotation by 270
        aug_3 = ds.map( lambda image, label: augment_image(image, label, rotate=True, num_rotations=3), tf.data.experimental.AUTOTUNE )
        
        # flip horizontal
        aug_4 = ds.map( lambda image, label: augment_image(image, label, horizontal_flip=True), tf.data.experimental.AUTOTUNE )
        
        # flip rot 90
        aug_5 = ds.map( lambda image, label: augment_image(image, label, rotate=True, num_rotations=1, horizontal_flip=True), tf.data.experimental.AUTOTUNE )
        
        # flip rot 180
        aug_6 = ds.map( lambda image, label: augment_image(image, label, rotate=True, num_rotations=2, horizontal_flip=True), tf.data.experimental.AUTOTUNE )

        # flip rot 270
        aug_7 = ds.map( lambda image, label: augment_image(image, label, rotate=True, num_rotations=3, horizontal_flip=True), tf.data.experimental.AUTOTUNE )

        ds = (ds.concatenate(aug_1)
                .concatenate(aug_2)
                .concatenate(aug_3)
                .concatenate(aug_4)
                .concatenate(aug_5)
                .concatenate(aug_6)
                .concatenate(aug_7))
        
    elif augment=="Hosseinzadeh":
        
        ds_size = ds_size*5
        
        # horizontal flip
        aug_1 = ds.map( lambda image, label: augment_image(image, label, horizontal_flip=True), tf.data.experimental.AUTOTUNE )
        
        # vertical flip
        aug_2 = ds.map( lambda image, label: augment_image(image, label, vertical_flip=True), tf.data.experimental.AUTOTUNE )
        
        # random brightness
        aug_3 = ds.map( lambda image, label: augment_image(image, label, brightness=True), tf.data.experimental.AUTOTUNE )
        
        # random contrast
        aug_4 = ds.map( lambda image, label: augment_image(image, label, contrast=True), tf.data.experimental.AUTOTUNE )
        
        ds = (ds.concatenate(aug_1)
                .concatenate(aug_2)
                .concatenate(aug_3)
                .concatenate(aug_4))

    return ds, ds_size