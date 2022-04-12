from src.preprocessing import read_test_csv_to_dataset, read_HAM10000_csv_to_dataset, rescale_and_resize, rescale_2
import pandas as pd
from tensorflow import keras
import os

from math import ceil

def evaluate_model(model, dataset, num_epochs, augmentation, img_width, img_height):
    
    if dataset=="ISIC":
        # Load file paths of test images to a tf dataset
        test_ds, test_size, y_true = read_test_csv_to_dataset()
        
        # Batch size the same as training set
        batch_size = 32
        test_steps = ceil(test_size / batch_size)
        
        # Convert filepaths to images and resize
        '''test, test_size = rescale_and_resize(ds=test_ds,
                                ds_size=test_size,
                                batch_size=batch_size,
                                training_set=False,
                                augment=augmentation,
                                img_width=img_width,
                                img_height=img_height)'''
        test, test_size = rescale_2(test_ds, 
                                    test_size, 
                                    batch_size, 
                                    False, 
                                    augmentation, 
                                    img_width, 
                                    img_height)
        
        # Evaluate model and save results in csv file
        metrics_dict = model.evaluate(test, return_dict=True) # dict with keys-metrics, values=metric vals
        
        # Add extra information
        metrics_dict["run_id"] = model.name
        metrics_dict["Epochs"] = num_epochs
        metrics_dict["Batch_Size"] = batch_size
        metrics_dict["Augmentation"] = augmentation
        metrics_dict["Image_Resolution"] = f"{img_width}x{img_height}"
        
        metrics_df = pd.DataFrame(metrics_dict, index=[0])
        
        metrics_df.to_csv(f"./output/results/{model.name}/test_scores.csv")
        
    elif dataset=="HAM10000":
        # Load file paths of test images to a tf dataset
        train_ds, train_size, test, test_size = read_HAM10000_csv_to_dataset()
        
        # Batch size the same as training set
        batch_size = 32
        test_steps = ceil(test_size / batch_size)
        
        # Convert filepaths to images and resize
        test, test_size = rescale_and_resize(ds=test_ds,
                                ds_size=test_size,
                                batch_size=batch_size,
                                training_set=False,
                                augment=augmentation,
                                img_width=img_width,
                                img_height=img_height)
        
        # Evaluate model and save results in csv file
        metrics_dict = model.evaluate(test, return_dict=True) # dict with keys-metrics, values=metric vals
        
        # Add extra information
        metrics_dict["run_id"] = model.name
        metrics_dict["Epochs"] = num_epochs
        metrics_dict["Augmentation"] = augmentation
        metrics_dict["Image_Resolution"] = f"{img_width}x{img_height}"
        
        metrics_df = pd.DataFrame(metrics_dict, index=[0])
        
        metrics_df.to_csv(f"./output/results/{model.name}/test_scores.csv")
    
#if __name__ == '__main__':
    
#    model = ResNet50_Mahbod("Mahbod_BL_Run_1")
#    model.load_weights("./output/training_ckpts/cp.ckpt")
#    evaluate_model(model, "now")