"""
Created by: Daniel Gallagher
Date: 7 March 2022
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
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


def read_HAM10000_csv_to_dataset2():
    
    # Read in filepaths and labels
    train_df = pd.read_csv("./metadata/train.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    # split into nevus and non nevus images
    train_other_df = train_df[ train_df["nv"]==0 ]
    test_other_df = test_df[ test_df["nv"]==0 ]
    train_nev_df = train_df[ train_df["nv"]==1 ]
    test_nev_df = test_df[ test_df["nv"]==1 ]
    
    # undersample the nevus images
    train_nev_df = train_nev_df.iloc[:1111]
    test_nev_df = test_nev_df.iloc[:1111]
    
    # Record dataset sizes for later use
    train_other_size = train_other_df.shape[0]
    test_other_size = test_other_df.shape[0]
    train_nev_size = train_nev_df.shape[0]
    test_nev_size = test_nev_df.shape[0]

    
    # Extract filepaths for creating tf dataset
    train_other_image_paths = "./images/HAM10000/" + train_other_df["image_id"] + ".jpg"
    test_other_image_paths = "./images/HAM10000/" + test_other_df["image_id"] + ".jpg"
    train_nev_image_paths = "./images/HAM10000/" + train_nev_df["image_id"] + ".jpg"
    test_nev_image_paths = "./images/HAM10000/" + test_nev_df["image_id"] + ".jpg"
    
    # Extract OneHot Labels for creating tf dataset
    train_other_labels = train_other_df.iloc[:,1:] 
    test_other_labels = test_other_df.iloc[:,1:] 
    train_nev_labels = train_nev_df.iloc[:,1:] 
    test_nev_labels = test_nev_df.iloc[:,1:] 
    
    # Create tf dataset objects
    train_other_ds = tf.data.Dataset.from_tensor_slices( (train_other_image_paths.values, train_other_labels.values) )
    test_other_ds = tf.data.Dataset.from_tensor_slices( (test_other_image_paths.values, test_other_labels.values) )
    train_nev_ds = tf.data.Dataset.from_tensor_slices( (train_nev_image_paths.values, train_nev_labels.values) )
    test_nev_ds = tf.data.Dataset.from_tensor_slices( (test_nev_image_paths.values, test_nev_labels.values) )
    
    return train_other_ds, train_other_size, test_other_ds, test_other_size, train_nev_ds, train_nev_size, test_nev_ds, test_nev_size
# ====================================================================================================



def map_decorator(func):
    def wrapper(steps, times, values):
        # Use a tf.py_function to prevent auto-graph from compiling the method
        return tf.py_function(
            func,
            inp=(steps, times, values),
            Tout=(steps.dtype, times.dtype, values.dtype)
        )
    return wrapper

#@tf.py_function(inp=[img], Tout=tf.float32)  
def white_balance(img, label): # source: https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
        # input image is RGB and tensor, convert to LAB for balancing
        result = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        # convert balanced image back to RGB
        result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
        result = tf.convert_to_tensor(result, dtype=tf.float32)
        return result, label
# ====================================================================================================

  
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
    
    #image = white_balance(image)
    
    # ImageNet mean RGB intensity subtraction
    #imagenet_rgb_mean = tf.reshape( tf.constant([0.485, 0.456, 0.406], dtype=tf.float32), [1,1,3]) 
    #image = image - imagenet_rgb_mean
    
    # Standardize image
    image = tf.image.per_image_standardization(image)

    return image, label

def mahdbod_preprocess_augment_dataset(ds, ds_size, batch_size, training_set, augment, img_width, img_height):

    if training_set:
        # Load image data from filepaths
        ds = ds.map(lambda feature, label: rescale_and_resize_image(feature, label, width=img_width, height=img_height), tf.data.experimental.AUTOTUNE)
        # Apply augmentation to increase dataset size
        ds, ds_size = augment_dataset(ds, ds_size, augment)
        # Cache augmented data
        ds = ds.cache()
        # Shuffle and repeat the augmented dataset for use in multiple epochss
        ds = ds.shuffle(buffer_size=ds_size, seed=42, reshuffle_each_iteration=True).repeat()
        # batch and prefetch
        ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        # Load image data from filepaths
        ds = ds.map(lambda feature, label: rescale_and_resize_image(feature, label, width=img_width, height=img_height), tf.data.experimental.AUTOTUNE)
        # Apply augmentation
        ds, ds_size = augment_dataset(ds, ds_size, augment)
        # batch and prefetch (no repeat needed for test set)
        ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds, ds_size


def hoss_preprocess_augment_dataset(ds_other, ds_other_size, ds_nev, ds_nev_size,
                                    batch_size, training_set, augment, img_width, img_height):
    
    # Train and val sets are cached, shuffled and repeated
    if training_set:
        # Load image data from filepaths of nevus and non nevus images
        ds_other = ds_other.map(lambda feature, label: rescale_and_resize_image(feature, label, width=img_width, height=img_height), tf.data.experimental.AUTOTUNE)
        ds_nev = ds_nev.map(lambda feature, label: rescale_and_resize_image(feature, label, width=img_width, height=img_height), tf.data.experimental.AUTOTUNE)
        
        # Apply augmentation to increase dataset size of non nevus images
        ds_other, ds_other_size = augment_dataset(ds_other, ds_other_size, augment)
        
        # Combine nevus and augmented non-nevus images to form balanced dataset
        ds = ds_other.concatenate(ds_nev)
        ds_size = ds_other_size + ds_nev_size
        
        # Cache augmented data
        ds = ds.cache()
        # Shuffle and repeat the augmented dataset for use in multiple epochss
        ds = ds.shuffle(buffer_size=ds_size, seed=42, reshuffle_each_iteration=True).repeat()
        # batch and prefetch
        ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    # test set does not need to bee cached, shuffled or repeated
    else:
        # Load image data from filepaths of nevus and non nevus images
        ds_other = ds_other.map(lambda feature, label: rescale_and_resize_image(feature, label, width=img_width, height=img_height), tf.data.experimental.AUTOTUNE)
        ds_nev = ds_nev.map(lambda feature, label: rescale_and_resize_image(feature, label, width=img_width, height=img_height), tf.data.experimental.AUTOTUNE)
        
        # Apply augmentation to increase dataset size of non nevus images
        ds_other, ds_other_size = augment_dataset(ds_other, ds_other_size, augment)
        
        # Combine nevus and augmented non-nevus images to form balanced dataset
        ds = ds_other.concatenate(ds_nev)
        ds_size = ds_other_size + ds_nev_size
        
        # batch and prefetch (no repeat needed for test set)
        ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds, ds_size


def rescale_and_resize(ds, ds_size, batch_size, training_set, augment, img_width, img_height):
    """Maps the rescale_and_resize_image function to the dataset."""
    
    # For train/val sets, adds repeat method
    if training_set:
        
        if augment=="Mahbod" or augment=="Hosseinzadeh":
        
            # Map image preprocessing and augmentation to dataset.
            #ds = ds.shuffle(buffer_size=ds_size, seed=42, reshuffle_each_iteration=True).repeat() 
            
            # Load image data from filepaths
            ds = ds.map(lambda feature, label: rescale_and_resize_image(feature, label, width=img_width, height=img_height))
            
            # Augment dataset to increase size
            ds, ds_size = augment_dataset(ds, ds_size, augment)
            
            # Cache the augmented dataset for reuse
            #ds = ds.cache()
            
            # Repeat then shuffle, batch and prefetch
            ds = (ds
                    .shuffle(buffer_size=ds_size, reshuffle_each_iteration=True, seed=42)
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



def run_preprocessing(batch_size, augment, dataset_name, img_width, img_height):
    """
    Returns training validation split as TF dataset objects. 
    Features are numpy arrays representing skin lesion images.
    Labels are OneHotEnc vectors. 
    """
    if dataset_name == "ISIC":
        # For batching the tf dataset objects
        
        
        train, train_size, val, val_size = create_train_val_tf_dataset()
        
        #train, train_size = rescale_and_resize(train, train_size, batch_size, training_set=True, augment=augment, img_width=img_width, img_height=img_height)
        #val, val_size = rescale_and_resize(val, val_size, batch_size, training_set=True, augment=augment, img_width=img_width, img_height=img_height)
        train, train_size = mahdbod_preprocess_augment_dataset(train,
                                                               train_size,
                                                               batch_size,
                                                               training_set=True,
                                                               augment=augment,
                                                               img_width=img_width,
                                                               img_height=img_height)
        
        val, val_size = mahdbod_preprocess_augment_dataset(val,
                                                           val_size,
                                                           batch_size,
                                                           training_set=True,
                                                           augment=augment,
                                                           img_width=img_width,
                                                           img_height=img_height)
        
        
    elif dataset_name == "HAM10000":
        
        """# load filepath datasets, use the test set as a validation set 
        train, train_size, val, val_size = read_HAM10000_csv_to_dataset()
        
        train, train_size = hoss_preprocess_augment_dataset(train,
                                                            train_size, 
                                                            batch_size,
                                                            training_set=True,
                                                            augment=augment,
                                                            img_width=img_width,
                                                            img_height=img_height)
        val, val_size = hoss_preprocess_augment_dataset(val,
                                                        val_size,
                                                        batch_size,
                                                        training_set=True, 
                                                        augment=augment, 
                                                        img_width=img_width,
                                                        img_height=img_height)"""

        # load filepath datasets of nevus and other classes
        train_other, train_other_size, val_other, val_other_size, train_nev, train_nev_size, val_nev, val_nev_size = read_HAM10000_csv_to_dataset2()
        
        # augment non nevus images and combine with nevus images to form final dataset
        train, train_size = hoss_preprocess_augment_dataset(train_other,
                                                            train_other_size, 
                                                            train_nev,
                                                            train_nev_size,
                                                            batch_size,
                                                            training_set=True,
                                                            augment=augment,
                                                            img_width=img_width,
                                                            img_height=img_height)
        
        val, val_size = hoss_preprocess_augment_dataset(val_other,
                                                        val_other_size,
                                                        val_nev,
                                                        val_nev_size, 
                                                        batch_size,
                                                        training_set=True,
                                                        augment=augment,
                                                        img_width=img_width,
                                                        img_height=img_height)
        
        
        
        
    return train, train_size, val, val_size
    

if __name__ == '__main__':
    read_HAM10000_csv_to_dataset2()