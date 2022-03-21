"""
Created by: Daniel Gallagher
Date: 7 March 2022
"""
import matplotlib
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def create_tf_dataset():
    """
    Reads in metadata, generates filepaths and creates tensorflow dataset objects.
    
    Returns TF dataset with features as filepaths, labels as OneHotEnc vectors
    """
    
    # Read metadata and create column with filepaths
    train_df = pd.read_csv("./metadata/train.csv")
    train_df["image_path"] = "./images/train/" + train_df["image_id"] + ".jpg"
    train_image_paths = train_df["image_path"]
    
    # Extract OneHot Labels
    train_labels = train_df.iloc[:,1:4] 
    
    # Create dataset object with features = filepaths, labels = OneHot vectors
    train_ds = tf.data.Dataset.from_tensor_slices( (train_image_paths.values, train_labels.values) )
    
    
    # Repeat above process for validation set
    val_df = pd.read_csv("./metadata/val.csv")
    val_df["image_path"] = "./images/val/" + val_df["image_id"] + ".jpg"
    val_image_paths = val_df["image_path"]
    val_labels = val_df.iloc[:,1:4] 
    
    val_ds = tf.data.Dataset.from_tensor_slices( (val_image_paths.values, val_labels.values) )
    
    
    return train_ds, val_ds
    
    

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

    return image, label

'''
def rescale_and_resize(ds, training_set):
    """Maps the rescale_and_resize_image function to the dataset."""

    # Map dataset.
    ds = ds.map(lambda feature, label: rescale_and_resize_image(feature, label, width=224, height=224))
    
    if training_set:
        ds = (ds
                .shuffle(buffer_size=128, seed=42) # size of dataset for perfect shuffling, set a seed here later
                .batch(32)
                .repeat()
                )
    else:
        ds = (ds
                .batch(150)
                .repeat()
                )
    
    return ds
'''
def rescale_and_resize(ds, training_set):
    """Maps the rescale_and_resize_image function to the dataset."""

    # Map dataset.
    ds = ds.map(lambda feature, label: rescale_and_resize_image(feature, label, width=224, height=224) , num_parallel_calls=tf.data.AUTOTUNE)
    
    
    ds = (ds
            .shuffle(buffer_size=128, seed=42) # size of dataset for perfect shuffling, set a seed here later
            .repeat()
            .batch(32)
            )
    
    return ds


def run_preprocessing():
    
    """
    Returns training validation split as TF dataset objects. 
    Features are numpy arrays representing skin lesion images.
    Labels are OneHotEnc vectors. 
    """
    
    train, val = create_tf_dataset()
    
    train = rescale_and_resize(train, training_set=True)
    val = rescale_and_resize(val, training_set=False)

    return train, val
    
    
def train_val_split():
    """
    Create small train/val split from just the validation data for developing
    """
    
    # Repeat above process for validation set
    test_df = pd.read_csv("./metadata/val.csv")
    test_df["image_path"] = "./images/val/" + test_df["image_id"] + ".jpg"
    test_image_paths = test_df["image_path"]
    test_labels = test_df.iloc[:,1:4] 
    
    train_paths = test_image_paths.iloc[:135]
    train_labels = test_labels.iloc[ :135, :]
    
    val_paths = test_image_paths.iloc[135:]
    val_labels = test_labels.iloc[ 135:, :]
            
    train_ds = tf.data.Dataset.from_tensor_slices( (train_paths.values, train_labels.values) )
    val_ds = tf.data.Dataset.from_tensor_slices( (val_paths.values, val_labels.values) )
    
    train = rescale_and_resize(train_ds, training_set=True)
    val = rescale_and_resize(val_ds, training_set=False)

    return train, val

#ds = rescale_and_resize( create_tf_dataset() )

# To check data is read in correctly
#plt.imshow( next(iter(ds))[0][0,:,:,:] )   
#plt.show()
    