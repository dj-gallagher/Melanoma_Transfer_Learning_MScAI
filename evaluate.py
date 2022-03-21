from preprocessing import create_tf_dataset, rescale_and_resize_image
import tensorflow as tf
import numpy as np
from tensorflow import keras

from pprint import pprint
import matplotlib.pyplot as plt
import cv2

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
        #ds = ds.batch(1).repeat()
        pass
    
    return ds

def evaluate_model(model, dataset):
    
    #model.evaluate(dataset)
    predictions = []
    labels = []
    
    for feature, label in dataset:
        prediction = model.predict(feature.numpy().reshape((-1,224,224,3)))
        predictions.append(prediction)
    pprint(predictions)

if __name__ == '__main__':

    train, val = create_tf_dataset()

    val_data = rescale_and_resize(val, training_set=False)

    model = keras.models.load_model( "./output/models/ResNet50_Hosseinzadeh_et_al_ResNet50_Hosseinzadeh_et_al_20220309-191906" )

    evaluate_model(model, val_data)
    '''
    counter = 0
    for feature, label in val_data:
        cv2.imshow("blan", feature.numpy())
        cv2.waitKey()
        counter += 1
        if counter == 5:
            break
    '''