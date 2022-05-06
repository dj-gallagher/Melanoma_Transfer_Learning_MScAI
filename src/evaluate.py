from src.preprocessing import read_test_csv_to_dataset, read_HAM10000_csv_to_dataset2, rescale_and_resize, mahdbod_preprocess_augment_dataset, hoss_preprocess_augment_dataset
import pandas as pd

import matplotlib.pyplot as plt
from math import ceil
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np

def evaluate_model(model, dataset, batch_size, num_epochs, augmentation, img_width, img_height):
    
    if dataset=="ISIC":
        # Load file paths of test images to a tf dataset
        test_ds, test_size, y_true = read_test_csv_to_dataset()
        
        # Batch size the same as training set
        
        test_steps = ceil(test_size / batch_size)
        
        # Convert filepaths to images and resize
        test, test_size = mahdbod_preprocess_augment_dataset(test_ds, 
                                                            test_size, 
                                                            batch_size, 
                                                            False, 
                                                            augmentation, 
                                                            img_width, 
                                                            img_height)
        
        # Evaluate model and save results in csv file
        metrics_dict = model.evaluate(test, return_dict=True) # dict with keys-metrics, values=metric vals
        
        
        # confusion matrix 
        y_pred = model.predict(test) # get predicted labels
        y_pred = y_pred.argmax(axis=1) # convert to ints
        
        plt.figure()
        plt.grid(False)
        matrix = confusion_matrix(y_true, y_pred)
        matrix_plot = ConfusionMatrixDisplay(matrix,
                               display_labels=["mel", "seb_ker", "nevus"]).plot()
        plt.savefig(f"./output/results/{model.name}/conf_matrix.png")
        
        
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
        #train_ds, train_size, test, test_size = read_HAM10000_csv_to_dataset()
        train_other, train_other_size, test_other, test_other_size, train_nev, train_nev_size, test_nev, test_nev_size = read_HAM10000_csv_to_dataset2()
        
        
        # Convert filepaths to images and resize
        test, test_size = hoss_preprocess_augment_dataset(test_other,
                                                        test_other_size,
                                                        test_nev,
                                                        test_nev_size, 
                                                        batch_size,
                                                        training_set=False,
                                                        augment=augmentation,
                                                        img_width=img_width,
                                                        img_height=img_height)
        
        # number of test steps
        test_steps = ceil(test_size / batch_size)
        
        # Evaluate model and save results in csv file
        metrics_dict = model.evaluate(test, return_dict=True) # dict with keys-metrics, values=metric vals
        
        y_true = [] # get true labels
        for img, label in test.unbatch(): # unbatch so all test samples are iter'd through, not just first batch
            int_label = np.argmax(label.numpy()) # need int labels for confusion matrix
            y_true.append(int_label)
        
        # confusion matrix 
        y_pred = model.predict(test) # get predicted labels
        y_pred = y_pred.argmax(axis=1) # convert to ints
        
        plt.figure()
        plt.grid(False)
        matrix = confusion_matrix(y_true, y_pred)
        matrix_plot = ConfusionMatrixDisplay(matrix,
                               display_labels=["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]).plot()
        plt.savefig(f"./output/results/{model.name}/conf_matrix.png")
        
        
        # Add extra information
        metrics_dict["run_id"] = model.name
        metrics_dict["Epochs"] = num_epochs
        metrics_dict["Batch_Size"] = batch_size
        metrics_dict["Augmentation"] = augmentation
        metrics_dict["Image_Resolution"] = f"{img_width}x{img_height}"
        
        metrics_df = pd.DataFrame(metrics_dict, index=[0])
        
        metrics_df.to_csv(f"./output/results/{model.name}/test_scores.csv")
    
    
    
#if __name__ == '__main__':
    
#    model = ResNet50_Mahbod("Mahbod_BL_Run_1")
#    model.load_weights("./output/training_ckpts/cp.ckpt")
#    evaluate_model(model, "now")