import datetime
from src.preprocessing import run_preprocessing
from src.evaluate import evaluate_model
from src.model import ResNet50_Mahbod, ResNet50_Hosseinzadeh, ResNet152V2_Rahman, train_model, save_model


if __name__ == '__main__':
    
    run_id = ""
    num_epochs = 15
    augmentation = "" # Mahbod / Hosseinzadeh
    dataset = "ISIC" # ISIC / HAM10000
    
    # Create training and validation sets from metadata and images folder
    train, train_size, val, val_size = run_preprocessing(augment=augmentation, dataset_name=dataset)
    
    # Create a model, pass run id as arguement
    #model = ResNet50_Hosseinzadeh(run_id)
    model = ResNet50_Mahbod(run_id)
    #model = ResNet152V2_Rahman(run_id)
    
    # To mark when training began, used for saving the model at the end of training
    training_start_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Train the model, logging training data with TensorBoard callback
    model = train_model(model, train, train_size, val, val_size, num_epochs)
    
    # Save the trained model
    #save_model(model, training_start_timestamp)
    
    # Find test set accuracy and save predictions
    evaluate_model(model, dataset, training_start_timestamp, num_epochs, augmentation)
    
    
