from src.preprocessing import read_test_csv_to_dataset, rescale_and_resize
import pandas as pd
from tensorflow import keras
import os

from math import ceil

def evaluate_model(model, timestamp):
    
    # Load file paths of test images to a tf dataset
    test_ds, test_size, y_true = read_test_csv_to_dataset()
    
    # Batch size the same as training set
    batch_size = 32
    test_steps = ceil(test_size / batch_size)
    
    # Convert filepaths to images and resize
    test, test_size = rescale_and_resize(ds=test_ds,
                              ds_size=test_size,
                              batch_size=batch_size,
                              training_set=False,
                              augment=False)
    
    # Evaluate model and save results in csv file
    metrics_dict = model.evaluate(test, return_dict=True)
    metrics_df = pd.DataFrame(metrics_dict, index=[0])
    
    os.mkdir(f"./output/results/{model.name}") # Directory for results
    
    metrics_df.to_csv(f"./output/results/{model.name}/results.csv")
    
#if __name__ == '__main__':
    
#    model = ResNet50_Mahbod("Mahbod_BL_Run_1")
#    model.load_weights("./output/training_ckpts/cp.ckpt")
#    evaluate_model(model, "now")