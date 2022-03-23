import datetime
from preprocessing import run_preprocessing
#from evaluate import evaluate_model
from model import ResNet50_Mahbod, ResNet50_Hosseinzadeh, ResNet152V2_Rahman, train_model, save_model


if __name__ == '__main__':
    
    # Create training and validation sets from metadata and images folder
    train, val = run_preprocessing()
    
    # Create a model
    #model = ResNet50_Hosseinzadeh()
    model = ResNet50_Mahbod()
    #model = ResNet152V2_Rahman()
    
    # Change model name for experiment to be run
    #model.name = "ResNet50_Mahbod_DifferentLR_Run_1"
    
    # To mark when training began, used for saving the model at the end of training
    training_start_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Train the model, logging training data with TensorBoard callback
    model = train_model(model, train, val)
    
    # Save the trained model
    save_model(model, training_start_timestamp)
    
    # Find test set accuracy and save predictions
    evaluate_model(model, training_start_timestamp)
    
    
