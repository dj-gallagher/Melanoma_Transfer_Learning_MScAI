import tensorflow as tf
from src.preprocessing import run_preprocessing
from src.evaluate import evaluate_model
from src.model import ResNet50_Mahbod, ResNet50_Hosseinzadeh, ResNet50, train_model

#from src.preprocessing import *
#from pprint import pprint
#import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    #tf.debugging.set_log_device_placement(True)
    
    with  tf.device("/cpu:0"):
        run_id = "TEST"
        num_epochs = 15
        augmentation = "Mahbod" # Mahbod / Hosseinzadeh
        dataset = "ISIC" # ISIC / HAM10000
        label_smooth_factor = 0
        img_width = 128
        img_height = 128
        
        # Create training and validation sets from metadata and images folder
        train, train_size, val, val_size = run_preprocessing(augment=augmentation, dataset_name=dataset, img_width=img_width, img_height=img_height)
    
        with tf.device("/gpu:0"):
            # Create a model, pass run id as arguement
            #model = ResNet50_Hosseinzadeh(run_id)
            model = ResNet50_Mahbod(run_id=run_id, label_smooth_factor=label_smooth_factor, img_width=img_width, img_height=img_height)
            #model = ResNet152V2_Rahman(run_id)
            #model = ResNet50(run_id=run_id, label_smooth_factor=label_smooth_factor, img_width=img_width, img_height=img_height)
            
            # Train the model, logging training data with TensorBoard callback
            trained_model = train_model(model, train, train_size, val, val_size, num_epochs)
            
            # Find test set accuracy and save predictions
            evaluate_model(trained_model, dataset, num_epochs, augmentation, img_width, img_height)    
'''
if __name__ == '__main__':
    #train_ds, train_size, val_ds, val_size = create_train_val_tf_dataset()
    
    test_ds, test_size, test_labels = read_test_csv_to_dataset()
    
    paths = []
    labels = []
    
    for path, label in test_ds:
        paths.append(path.numpy())
        labels.append(label.numpy())
    
    test_ds, test_size = rescale_and_resize(ds=test_ds,
                                            ds_size=test_size,
                                            batch_size=1,
                                            training_set=True,
                                            augment="None",
                                            img_width=128,
                                            img_height=128)
    
    counter = 0
    for image, label in test_ds.unbatch():
        plt.imshow(image)
        plt.show()
        if counter==0:
            break
'''
    
    