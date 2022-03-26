import datetime
from preprocessing import run_preprocessing
from evaluate import evaluate_model
from model import ResNet50_Mahbod, ResNet50_Hosseinzadeh, ResNet152V2_Rahman, train_model, save_model


if __name__ == '__main__':
    
    # Create training and validation sets from metadata and images folder
    train, train_size, val, val_size = run_preprocessing(augment=True)
    
    # Create a model, pass run id as arguement
    #model = ResNet50_Hosseinzadeh()
    model = ResNet50_Mahbod()
    #model = ResNet152V2_Rahman()
    
    # To mark when training began, used for saving the model at the end of training
    training_start_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Train the model, logging training data with TensorBoard callback
    model = train_model(model, train, train_size, val, val_size, 1)
    
    # Save the trained model
    #save_model(model, training_start_timestamp)
    
    # Find test set accuracy and save predictions
    evaluate_model(model, training_start_timestamp)
    
    
