import tensorflow as tf
from tensorflow import keras
import datetime
import math
import tensorflow_addons as tfa
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('Agg') # https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined/3054314#3054314

# BASELINE MODEL FUNCTIONS
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

def ResNet50_Mahbod(model_name):
    """
    Creates a Keras model and from a base pre-trained model and newly defined output layers.
    Compiles the model with defined optimizer, loss and metrics.
    
    Returns: compiled Keras model ready for training
    """
        
    # DEFINING MODEL LAYERS
    # ---------------------------
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights="imagenet",
                                                      input_shape=(224,224,3))
    
    #base_model.trainable = False # Blocks 1-17 Frozen as in Mahbod et al.
    base_model.trainable = False # Blocks 1-17 Frozen as in Mahbod et al.
    
    
    # Define output layers (Mahbod et al. used here)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(units=64, activation="relu")(x)
    predictions = keras.layers.Dense(units=3, activation="softmax")(x)

    # Create model using forzen base layers and new FC layers
    model = keras.models.Model(inputs=base_model.input, 
                               outputs=predictions, 
                               name=model_name) 
    
    
    
    # UNFREEZE 17TH BLOCK
    # -------------------------------------
    # Create dictionary of layer name and whether the layer is trainable 
    trainable_dict = dict([ (layer.name, layer.trainable) for layer in model.layers ])
    
    # Identify names of layers in 17th block
    block_17_names = []
    for name in [ layer.name for layer in model.layers ]: # iterate through model layer names
        if "conv5_block3" in name: # conv5_block3 is naming schemee for 17th block
            block_17_names.append(name)
            
    # Set these layers to be trainable
    for name in block_17_names:
        trainable_dict[name] = True  # change dict values to true     
    
    for layer_name, trainable_bool in trainable_dict.items():
        layer = model.get_layer(name=layer_name)
        layer.trainable = trainable_bool
    

    # OPTIMIZERS
    # -------------------------------------
    # Different LR for pretrained and FC layers
    pretrained_lr = 0.0001 
    new_lr = 10 * pretrained_lr 
    
    # Set LR schedule
    num_epochs = 15
    steps_per_epoch = 16000 // 32 # train_size // batch_size
    
    step = tf.Variable(0, trainable=False)
    
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[steps_per_epoch*5, steps_per_epoch*10], # drop LR at epoch 5 and 10
        values=[0.001, 0.0001, 0.00001]) # drop by factor of 10 each time
    
    # lr and wd can be a function or a tensor
    #pretrained_lr = schedule(step)
    #new_lr = 10 * schedule(step)
    
    # Create multioptimizer -----
    optimizers = [keras.optimizers.Adam(learning_rate=pretrained_lr),
                  keras.optimizers.Adam(learning_rate=new_lr)]
    
    # Layer objects for pre-trained and FC layers
    block_17_layers = [ model.get_layer(name=name) for name in block_17_names ]
    new_fc_layers = model.layers[-3:]
    
    # (Optimizer, layer) pairs 
    block_17_optimizers_and_layers =  [(optimizers[0], block_17_layers)]  #[  (optimizers[0],layer) for layer in block_17_layers ]
    new_fc_optimizers_and_layers = [(optimizers[1], new_fc_layers)]  #[  (optimizers[1],layer) for layer in new_fc_layers ]
    optimizers_and_layers = block_17_optimizers_and_layers + new_fc_optimizers_and_layers
    
    # Optimizer with different learning rates across layers
    #optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    optimizer = keras.optimizers.Adam(learning_rate=pretrained_lr)
    # ---------------------------
    
    # LOSS FUNCTION AND METRICS
    # -------------------------------------
    loss_func = keras.losses.CategoricalCrossentropy()
    metrics_list = ['accuracy',
                    keras.metrics.AUC( multi_label=True )] 
    
    
    # COMPILE 
    # -------------------------------------
    model.compile(optimizer=optimizer ,
                loss=loss_func ,
                metrics=metrics_list)
    
    return model


def ResNet50_Hosseinzadeh(model_name):
    
    # DEFINING MODEL LAYERS
    # ---------------------------
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights="imagenet",
                                                      input_shape=(224,224,3))
    
    base_model.trainable = True 

    # Define output layers 
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(units=64, activation="relu")(x)
    x = keras.layers.Dropout(rate=0.5)(x)
    predictions = keras.layers.Dense(units=7, activation="softmax")(x)

    # Create model using forzen base layers and new FC layers
    model = keras.models.Model(inputs=base_model.input, 
                               outputs=predictions, 
                               name=model_name) 


    # OPTIMIZERS
    # -------------------------------------    
    optimizer = tfa.optimizers.AdamW(weight_decay=(math.e)**(-5),
                                     learning_rate=(math.e)**(-5),
                                     beta_1=0.9,
                                     beta_2=0.999,
                                     epsilon=(math.e)**(-8))
    
    # LOSS FUNCTION AND METRICS
    # -------------------------------------
    loss_func = keras.losses.CategoricalCrossentropy()
    metrics_list = ['accuracy',
                    keras.metrics.AUC( multi_label=True )] 
    
    
    # COMPILE 
    # -------------------------------------
    model.compile(optimizer=optimizer ,
                loss=loss_func ,
                metrics=metrics_list)
                

    return model
    

def ResNet152V2_Rahman(model_name):
    
    # DEFINING MODEL LAYERS
    # ---------------------------
    # Load pre trained model without last FC layer
    base_model = keras.applications.ResNet152V2(include_top=False,
                                                weights="imagenet",
                                                input_shape=(224,224,3))
    
    # Freeze all pre trained layers
    base_model.trainable = False
    
    # Define output layers (Rahman and Ami)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    predictions = keras.layers.Dense(units=3, activation="softmax")(x)

    # Create model using forzen base layers and new FC layers
    model = keras.models.Model(inputs=base_model.input, 
                               outputs=predictions, 
                               name=model_name) 
    
    # COMPILING THE MODEL
    # ---------------------------
    lr = 0.001 # 10x all other layers
    #lr_schedule = [0.001, 0.0001, 0.00001] # drop by 10x factor at epoch 5 and 10

    optimiser = keras.optimizers.Adam(learning_rate=lr)
    loss_func = keras.losses.CategoricalCrossentropy()
    metrics_list = ['accuracy',
                    keras.metrics.AUC( multi_label=True )] 
    
    model.compile(optimizer=optimiser ,
                loss=loss_func ,
                metrics=metrics_list 
                )
    
    return model
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------


# CALLBACKS
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
def create_checkpoint_callback(model_name):
    # SAVE WEIGHTS DURING TRAINING
    checkpoint_path = f"./output/training_ckpts/{model_name}/cp.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_freq="epoch",
                                                  save_weights_only=True,
                                                  verbose=1)    
    
    return cp_callback


def create_tensorboard_callback(model_name):
    log_dir = "./output/logs/fit/" + model_name #+ "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,  
                                                       update_freq="epoch",
                                                       histogram_freq=1)
    
    return tensorboard_callback
'''
def create_lr_scheduler_cb():
    """
    Callback function to drop LR by factor of 10 at
    the 5th and 10th epoch
    """
    def scheduler(epoch, learning_rate):
        if epoch % 5 == 0:
            return learning_rate * 0.1
        else:
            return learning_rate
        
    cb = keras.callbacks.LearningRateScheduler(schedule=scheduler, verbose=1)
    
    return cb
'''
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

def save_training_plots(history, model_name, num_epochs):
    # Here we are going to append the training (accuracy + loss) and validation (accuracy + loss) to a 
    # 2D NumPy array and save to a file gpu_model_data.csv'
    #training_loss = np.array(model.history["loss"]).reshape((num_epochs, 1))
    #validation_loss = np.array(model.history["val_loss"]).reshape((num_epochs, 1))
    #training_accuracy = np.array(model.history["accuracy"]).reshape((num_epochs, 1))
    #validation_accuracy = np.array(model.history["val_accuracy"]).reshape((num_epochs, 1))
    #metric_data = training_data = np.hstack((training_accuracy,training_loss,validation_accuracy, validation_loss))
    #np.savetxt(f'./output/results/{model.name}_training_data.csv', metric_data, delimiter=',')


    # The following code will save an image showing the above metrics for the model during the training process. 

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, num_epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, num_epochs), history.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'./output/results/{model_name}_train_loss.png')
    
    plt.figure()
    plt.plot(np.arange(0, num_epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, num_epochs), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f'./output/results/{model_name}_train_accuracy.png')


# MODEL TRAINING 
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
def train_model(model, train, train_size, val, val_size, num_epochs):
    
    # Create directories to store training checkpoints
    #os.mkdir(f"./output/logs/fit/{model.name}") # tensorboard cb
    os.mkdir(f"./output/training_ckpts/{model.name}") # checkpoint cb
    
    # Create directory to store training and testing data
    os.mkdir(f"./output/results/{model.name}") # Directory for results
    
    # Values for ISIC 2017, will have to make this automatic later
    # Used to calcualte how many steps per epoch and per validation 
    batch_size = 32
    
    # Create list of callback functions
    checkpoint_cb = create_checkpoint_callback(model.name)
    #cb_tensorboard = create_tensorboard_callback(model.name)
    #cb_lr_schedule = create_lr_scheduler_cb()
    cb_list = [checkpoint_cb]
    
        
    history = model.fit(train, 
                epochs=num_epochs,
                steps_per_epoch=(train_size//batch_size), # should be a number s.t. (steps*batch_size)=num_training_egs
                validation_data=val, 
                validation_steps=(val_size//batch_size),
                callbacks=cb_list, 
                verbose=1) 
    
    save_training_plots(history, model.name, num_epochs)
    
    return model
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
    
    
# MODEL SAVING  
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
def save_model(trained_model, timestamp):
    
    #trained_model.save( "./output/models/" + f"{trained_model.name}_{timestamp}")
    trained_model.save( "./output/models/" + f"{trained_model.name}_{timestamp}", include_optimizer=False)
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------


    
    
  
    
        
