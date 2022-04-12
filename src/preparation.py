"""
Created by: Daniel Gallagher
Date: 7 March 2022    

Summary: Once the raw ISIC 2017 datasets have been downloaded, executing this file will create 3 csv files train, val and test.
         Each csv contains and image_id and a one hot encoded label for one of the 3 skin lesion types: Melanoma, Seborrheic Keratosis or Nevus

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def create_isic_2017_metadata():
    
    # TRAINING DATA
    # --------------------
    # Load original validation set metadata
    df = pd.read_csv("../metadata/ISIC-2017_Training_Part3_GroundTruth.csv")

    # Create third column to store nevus == True/False label
    df["nevus"] = pd.Series( np.zeros(len(df)) )
    
    # Label nevus images
    for ind in range(len(df)):
        if df.loc[ind, "melanoma"] == 0 and df.loc[ind, "seborrheic_keratosis"] == 0:
            df.loc[ind, "nevus"] = 1
    
    # Write resulting metadata to csv file
    df.to_csv("../metadata/train.csv", index=False)
    
    # VALIDATION METADATA
    # --------------------
    # Load original validation set metadata
    df = pd.read_csv("../metadata/ISIC-2017_Validation_Part3_GroundTruth.csv")

    # Create third column to store nevus == True/False label
    df["nevus"] = pd.Series( np.zeros(len(df)) )
    
    # Label nevus images
    for ind in range(len(df)):
        if df.loc[ind, "melanoma"] == 0 and df.loc[ind, "seborrheic_keratosis"] == 0:
            df.loc[ind, "nevus"] = 1
    
    # Write resulting metadata to csv file
    df.to_csv("../metadata/val.csv", index=False)
    
    # TEST METADATA
    # --------------------
    # Load original validation set metadata
    df = pd.read_csv("../metadata/ISIC-2017_Test_v2_Part3_GroundTruth.csv")

    # Create third column to store nevus == True/False label
    df["nevus"] = pd.Series( np.zeros(len(df)) )
    
    # Label nevus images
    for ind in range(len(df)):
        if df.loc[ind, "melanoma"] == 0 and df.loc[ind, "seborrheic_keratosis"] == 0:
            df.loc[ind, "nevus"] = 1
    
    # Write resulting metadata to csv file
    df.to_csv("./metadata/test.csv", index=False)
    

def create_HAM10000_metadata_splits():
    
    # Load original metadata file, onehot encode labels and create new dataframe with image_id, onehot label
    original_metadata_df = pd.read_csv("../metadata/HAM10000_metadata.csv") # use ../ to start from parent dir
    original_metadata_df = original_metadata_df[["image_id", "dx"]]

    one_hot_label_df = pd.get_dummies(original_metadata_df["dx"])
    metadata_df = pd.concat([original_metadata_df["image_id"], one_hot_label_df], axis=1)
    
    # Create train/test split. Hosseinzadeh did not use validation set
    train_df, test_df = train_test_split(metadata_df, test_size=0.3, random_state=42)
    
    train_df.to_csv("../metadata/train.csv", index=False)
    test_df.to_csv("../metadata/test.csv", index=False)
