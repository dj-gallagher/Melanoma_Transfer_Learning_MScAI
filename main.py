import tensorflow as tf
from src.preprocessing import run_preprocessing
from src.evaluate import evaluate_model
from src.model import ResNet50_Mahbod, ResNet50_Hosseinzadeh, ResNet50, train_model


if __name__ == '__main__':
    
    #tf.debugging.set_log_device_placement(True)
    
    with  tf.device("/cpu:0"):
        run_id = "TEST"
        num_epochs = 2
        augmentation = "None" # Mahbod / Hosseinzadeh
        dataset = "ISIC" # ISIC / HAM10000
        label_smooth_factor = 0
        img_width = 128
        img_height = 128
        
        # Create training and validation sets from metadata and images folder
        train, train_size, val, val_size = run_preprocessing(augment=augmentation, dataset_name=dataset, img_width=img_width, img_height=img_height)
    
        with tf.device("/gpu:0"):
            # Create a model, pass run id as arguement
            #model = ResNet50_Hosseinzadeh(run_id)
            #model = ResNet50_Mahbod(run_id=run_id, label_smooth_factor=label_smooth_factor, img_width=img_width, img_height=img_height)
            #model = ResNet152V2_Rahman(run_id)
            model = ResNet50(run_id=run_id, label_smooth_factor=label_smooth_factor, img_width=img_width, img_height=img_height)
            
            # Train the model, logging training data with TensorBoard callback
            model = train_model(model, train, train_size, val, val_size, num_epochs)
            
            # Find test set accuracy and save predictions
            evaluate_model(model, dataset, num_epochs, augmentation, img_width, img_height)    

