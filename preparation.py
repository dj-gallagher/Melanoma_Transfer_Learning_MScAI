"""
Created by: Daniel Gallagher
Date: 7 March 2022    

Summary: Once the raw ISIC 2017 datasets have been downloaded, executing this file will create 3 csv files train, val and test.
         Each csv contains and image_id and a one hot encoded label for one of the 3 skin lesion types: Melanoma, Seborrheic Keratosis or Nevus

"""

import pandas as pd
import numpy as np


def create_metadata():
    
    # TRAINING DATA
    # --------------------
    # Load original validation set metadata
    df = pd.read_csv("./metadata/ISIC-2017_Training_Part3_GroundTruth.csv")

    # Create third column to store nevus == True/False label
    df["nevus"] = pd.Series( np.zeros(len(df)) )
    
    # Label nevus images
    for ind in range(len(df)):
        if df.loc[ind, "melanoma"] == 0 and df.loc[ind, "seborrheic_keratosis"] == 0:
            df.loc[ind, "nevus"] = 1
    
    # Write resulting metadata to csv file
    df.to_csv("./metadata/train.csv", index=False)
    
    # VALIDATION METADATA
    # --------------------
    # Load original validation set metadata
    df = pd.read_csv("./metadata/ISIC-2017_Validation_Part3_GroundTruth.csv")

    # Create third column to store nevus == True/False label
    df["nevus"] = pd.Series( np.zeros(len(df)) )
    
    # Label nevus images
    for ind in range(len(df)):
        if df.loc[ind, "melanoma"] == 0 and df.loc[ind, "seborrheic_keratosis"] == 0:
            df.loc[ind, "nevus"] = 1
    
    # Write resulting metadata to csv file
    df.to_csv("./metadata/val.csv", index=False)
    
    # TEST METADATA
    # --------------------
    # Load original validation set metadata
    df = pd.read_csv("./metadata/ISIC-2017_Test_v2_Part3_GroundTruth.csv")

    # Create third column to store nevus == True/False label
    df["nevus"] = pd.Series( np.zeros(len(df)) )
    
    # Label nevus images
    for ind in range(len(df)):
        if df.loc[ind, "melanoma"] == 0 and df.loc[ind, "seborrheic_keratosis"] == 0:
            df.loc[ind, "nevus"] = 1
    
    # Write resulting metadata to csv file
    df.to_csv("./metadata/test.csv", index=False)
    
    
    

if __name__ == '__main__':
    create_metadata()