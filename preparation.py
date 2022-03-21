"""
Created by: Daniel Gallagher
Date: 7 March 2022    
"""

import pandas as pd
import numpy as np


def create_metadata():
    
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
    
    
    

if __name__ == '__main__':
    create_metadata()