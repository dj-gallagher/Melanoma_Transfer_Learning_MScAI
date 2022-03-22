from sympy import primitive
from preprocessing import read_test_csv_to_dataset, rescale_and_resize
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf

from math import ceil
from sklearn.metrics import accuracy_score

def evaluate_model(model):

    test_ds, test_size, y_true = read_test_csv_to_dataset()
    
    batch_size = 32
    test_steps = ceil(test_size / batch_size)
    
    test = rescale_and_resize(ds=test_ds,
                              ds_size=test_size,
                              batch_size=32,
                              training_set=False)

    y_pred = model.predict(test,
                steps=test_steps,
                verbose=1)
    
    np.savetxt(fname="./output/results/test_predictions.csv",
               X=y_pred,
               delimiter=",")
    
    #df = pd.read_csv("./output/results/test_predictions.csv", header=None)
    #y_pred = df.to_numpy()
    #y_pred = (y_pred > 0.5)
    
    print("\n" , "TEST SET ACCURACY: ")
    print( "\t-> " , str(accuracy_score(y_true, y_pred)) )
    