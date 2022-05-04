import tensorflow as tf
from src.preprocessing import run_preprocessing
from src.evaluate import evaluate_model
from src.model import *
from src.preprocessing import *
from src.improvements import *

import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    
    with  tf.device("/gpu:0"):
        run_id = "Mahbod_LS_6"
        EPOCHS = 15
        BATCH_SIZE = 64
        AUGMENTATION = "Mahbod" # Mahbod / Hosseinzadeh
        DATASET = "ISIC" # ISIC / HAM10000
        LABEL_SMOOTHING = 0.1
        IMG_WIDTH = 128
        IMG_HEIGHT = 128
        LR = 0.001
        DROPOUT_RATE = 0
        LR_SCHEDULE = False
        WEIGHT_DECAY = 0
        MIN_LR_FACTOR = 0
        
        # Create training and validation sets from metadata and images folder
        train, train_size, val, val_size = run_preprocessing(batch_size=BATCH_SIZE,
                                                             augment=AUGMENTATION,
                                                             dataset_name=DATASET,
                                                             img_width=IMG_WIDTH,
                                                             img_height=IMG_HEIGHT)
        print("\n")
        print("TRAINING SET SIZE = ", train_size)
        print("VALIDATION SET SIZE = ", val_size)
        print("\n")
        
        # Create a model, pass run id as arguement
        # ----------------------------------------
        #model = ResNet50_Mahbod(run_id=run_id, label_smooth_factor=LABEL_SMOOTHING, 
        #                       img_width=IMG_WIDTH, img_height=IMG_HEIGHT, lr=LR)
        
        #model = Mahbod_ResNet50_Dropout(run_id=run_id, label_smooth_factor=LABEL_SMOOTHING, 
        #                                  img_width=IMG_WIDTH, img_height=IMG_HEIGHT, 
        #                                   lr=LR, dropout_rate=DROPOUT_RATE, train_size=train_size, 
        #                                   batch_size=BATCH_SIZE, num_epochs=EPOCHS)
        
        model = Mahbod_Resnet50_CosineLRDecay(run_id=run_id, label_smooth_factor=LABEL_SMOOTHING, 
                                          img_width=IMG_WIDTH, img_height=IMG_HEIGHT, 
                                           lr=LR, dropout_rate=DROPOUT_RATE, train_size=train_size, 
                                           batch_size=BATCH_SIZE, num_epochs=EPOCHS)
        
        #model = ResNet50_Hosseinzadeh(run_id=run_id, 
        #                            label_smooth_factor=LABEL_SMOOTHING,
        #                            img_width=IMG_WIDTH, 
        #                            img_height=IMG_HEIGHT, 
        #                            lr=LR, 
        #                            dropout_rate=DROPOUT_RATE,
        #                            weight_decay=WEIGHT_DECAY)
        
        #model = Hosseinzadeh_ResNet50_CosineLRDecay(run_id=run_id, 
        #                                            label_smooth_factor=0,
        #                                            img_width=225, 
        #                                            img_height=300, 
        #                                            lr=(math.e)**(-5), 
        #                                            dropout_rate=0.5,
        #                                            weight_decay=(math.e)**(-5),
        #                                            min_lr_factor=MIN_LR_FACTOR,
        #                                           train_size=train_size,
        #                                            batch_size=BATCH_SIZE,
        #                                            num_epochs=EPOCHS) 
        # ----------------------------------------      
        
        # Train the model
        trained_model = train_model(model,
                                    train, 
                                    train_size,
                                    val,
                                    val_size,
                                    EPOCHS,
                                    BATCH_SIZE,
                                    lr_schedule=LR_SCHEDULE)
        
        # Find test set accuracy and save predictions
        evaluate_model(trained_model, DATASET, BATCH_SIZE, EPOCHS, AUGMENTATION, IMG_WIDTH, IMG_HEIGHT)  


'''if __name__ == '__main__':
    #train_ds, train_size, val_ds, val_size = create_train_val_tf_dataset()
    
    test_ds, test_size, test_labels = read_test_csv_to_dataset()
    
    test_ds, test_size = rescale_and_resize(ds=test_ds,
                                            ds_size=test_size,
                                            batch_size=1,
                                            training_set=True,
                                            augment="None",
                                            img_width=128,
                                            img_height=128)
    
    #final = np.hstack((img, white_balance_loops(img)))
    #show(final)
    #cv.imwrite('result.jpg', final)
    
    counter = 0
    for image, label in test_ds.unbatch():
        counter+=1
        
        plt.imshow(image.numpy())
        plt.show()
        
        if counter==1:
            break'''
    
    