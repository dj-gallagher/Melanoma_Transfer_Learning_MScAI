"""
Created by: Daniel Gallagher
Date: 7 March 2022
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.augment import augment_dataset

def create_train_val_tf_dataset():
    """
    Reads in metadata, generates filepaths and creates tensorflow dataset objects.
    
    Returns TF dataset with features as filepaths, labels as OneHotEnc vectors
    """
    
    # Read metadata csv and create column with filepaths
    train_df = pd.read_csv("./metadata/train.csv")
    train_df["image_path"] = "./images/train/" + train_df["image_id"] + ".jpg"
    train_image_paths = train_df["image_path"]
    
    # Number of training examples
    train_size = train_df.shape[0]
    
    # Extract OneHot Labels
    train_labels = train_df.iloc[:,1:4] 
    
    # Create dataset object with features = filepaths, labels = OneHot vectors
    train_ds = tf.data.Dataset.from_tensor_slices( (train_image_paths.values, train_labels.values) )
    
    
    # Repeat above process for validation set
    val_df = pd.read_csv("./metadata/val.csv")
    val_df["image_path"] = "./images/val/" + val_df["image_id"] + ".jpg"
    val_image_paths = val_df["image_path"]
    val_labels = val_df.iloc[:,1:4] 
    val_size = val_df.shape[0]
    
    val_ds = tf.data.Dataset.from_tensor_slices( (val_image_paths.values, val_labels.values) )
    
    
    return train_ds, train_size, val_ds, val_size

def read_test_csv_to_dataset():  
    
    # Read metadata csv and create column with filepaths
    test_df = pd.read_csv("./metadata/test.csv")
    test_df["image_path"] = "./images/test/" + test_df["image_id"] + ".jpg"
    test_image_paths = test_df["image_path"]
    
    # Number of test examples
    test_size = test_df.shape[0]
    
    # Extract OneHot Labels
    test_labels = test_df.iloc[:,1:4] 
    
    # Create dataset object with features = filepaths, labels = OneHot vectors
    test_ds = tf.data.Dataset.from_tensor_slices( (test_image_paths.values, test_labels.values) )

    return test_ds, test_size, test_labels.values

def read_HAM10000_csv_to_dataset():
    
    # Read in filepaths and labels
    train_df = pd.read_csv("./metadata/train.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    # Record dataset sizes for later use
    train_size = train_df.shape[0]
    test_size = test_df.shape[0]

    # Extract filepaths for creating tf dataset
    train_image_paths = "./images/HAM10000/" + train_df["image_id"] + ".jpg"
    test_image_paths = "./images/HAM10000/" + test_df["image_id"] + ".jpg"
    
    # Extract OneHot Labels for creating tf dataset
    train_labels = train_df.iloc[:,1:] 
    test_labels = test_df.iloc[:,1:] 
    
    # Create tf dataset objects
    train_ds = tf.data.Dataset.from_tensor_slices( (train_image_paths.values, train_labels.values) )
    test_ds = tf.data.Dataset.from_tensor_slices( (test_image_paths.values, test_labels.values) )
    
    return train_ds, train_size, test_ds, test_size
    

def rescale_and_resize_image(file_path, label, width, height): 
    """
    Opens, resizes and rescales the image associated with the file_path supplied.

    All images in all datasets need to be resized and rescaled for inputting to the model.

    :param file_path: The path to the image file to open, resize and rescale.
    :param label: The label associated with the image.
    :param conf: The flow configuration.
    :return: The resized and rescaled image and its label.
    """
    # Read the image data from disk.
    image = tf.io.read_file(file_path)
    # Convert to JPEG
    image = tf.image.decode_jpeg(image, channels=3)
    # Convert from [0, 255] -> [0.0, 1.0]
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize shape to model input dimensions
    image = tf.image.resize(image, [width, height])
    
    # ImageNet mean RGB intensity subtraction
    #imagenet_rgb_mean = tf.reshape( tf.constant([0.485, 0.456, 0.406], dtype=tf.float32), [1,1,3]) 
    #image = image - imagenet_rgb_mean
    
    # Standardize image
    #image = tf.image.per_image_standardization(image)

    return image, label


def rescale_and_resize(ds, ds_size, batch_size, training_set, augment, img_width, img_height):
    """Maps the rescale_and_resize_image function to the dataset."""
    
    # For train/val sets, adds repeat method
    if training_set:
        
        if augment=="Mahbod" or augment=="Hosseinzadeh":
        
            # Map image preprocessing and augmentation to dataset.
            #ds = ds.shuffle(buffer_size=ds_size, seed=42).repeat() 
            
            # Load image data from filepaths
            ds = ds.map(lambda feature, label: rescale_and_resize_image(feature, label, width=img_width, height=img_height))
            
            # Augment dataset to increase size
            ds, ds_size = augment_dataset(ds, ds_size, augment)
            
            # Cache the augmented dataset for reuse
            ds = ds.cache()
            
            # Repeat then shuffle, batch and prefetch
            ds = (ds
                    .shuffle(100, reshuffle_each_iteration=True)
                    .repeat() 
                    #.shuffle(100, reshuffle_each_iteration=True)
                    .batch(batch_size)
                    .prefetch(100)
                    )
        
        else:
            # Shuffle and repeat the dataset when it is just filepaths first, otherwise shuffling would be 
            # applied to loaded images which takes long
            ds = (ds
                    .shuffle(buffer_size=ds_size, seed=42) 
                    .repeat()
                    )
            
            # Map image preprocessing/augmentation to dataset.
            ds = ds.map(lambda feature, label: rescale_and_resize_image(feature, label, width=img_width, height=img_height))
            
            ds = (ds
                    .batch(batch_size)
                    .prefetch(100)
                    )
    else:
        if augment=="Mahbod" or augment=="Hosseinzadeh":
            # Map image preprocessing/augmentation to dataset.
            ds = ds#.shuffle(buffer_size=ds_size, seed=42) 
            ds = ds.map(lambda feature, label: rescale_and_resize_image(feature, label, width=img_width, height=img_height))
            ds, ds_size = augment_dataset(ds, ds_size, augment)
            ds = (ds
                    .batch(batch_size)
                    .prefetch(100)
                    )
        else:
            # Map image preprocessing/augmentation to dataset.
            ds = ds.shuffle(buffer_size=ds_size, seed=42) 
            ds = ds.map(lambda feature, label: rescale_and_resize_image(feature, label, width=img_width, height=img_height))
            
            ds = (ds
                    .batch(batch_size)
                    .prefetch(100)
                    )
    
    return ds, ds_size



def run_preprocessing(augment, dataset_name, img_width, img_height):
    """
    Returns training validation split as TF dataset objects. 
    Features are numpy arrays representing skin lesion images.
    Labels are OneHotEnc vectors. 
    """
    if dataset_name == "ISIC":
        # For batching the tf dataset objects
        batch_size = 32
        
        train, train_size, val, val_size = create_train_val_tf_dataset()
        
        train, train_size = rescale_and_resize(train, train_size, batch_size, training_set=True, augment=augment, img_width=img_width, img_height=img_height)
        val, val_size = rescale_and_resize(val, val_size, batch_size, training_set=True, augment=augment, img_width=img_width, img_height=img_height)
    
    elif dataset_name == "HAM10000":
        # For batching the tf dataset objects
        batch_size = 32
        
        train, train_size, val, val_size = read_HAM10000_csv_to_dataset()
        
        train, train_size = rescale_and_resize(train, train_size, batch_size, training_set=True, augment=augment, img_width=img_width, img_height=img_height)
        val, val_size = rescale_and_resize(val, val_size, batch_size, training_set=True, augment=augment, img_width=img_width, img_height=img_height)
    
    return train, train_size, val, val_size
    
    