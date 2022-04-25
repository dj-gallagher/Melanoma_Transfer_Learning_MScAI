import tensorflow as tf
from tensorflow import keras
import datetime
import math
import tensorflow_addons as tfa
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#from keras_lr_multiplier import LRMultiplier

matplotlib.use('Agg') # https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined/3054314#3054314

# MAHBOD
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

def ResNet50_Mahbod(run_id,
                    label_smooth_factor=0,
                    img_width=224,
                    img_height=224,
                    lr=0.0001):
    """
    Creates a Keras model and from a base pre-trained model and newly defined output layers.
    Compiles the model with defined optimizer, loss and metrics.
    
    Returns: compiled Keras model ready for training
    """
        
    # DEFINING MODEL LAYERS
    # ---------------------------
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights="imagenet",
                                                      input_shape=(img_width,img_height,3))
    
    #base_model.trainable = False # Blocks 1-17 Frozen as in Mahbod et al.
    base_model.trainable = False # Blocks 1-17 Frozen as in Mahbod et al.
    
    
    # Define output layers (Mahbod et al. used here)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(units=64, 
                           activation="relu", 
                           kernel_initializer=keras.initializers.RandomNormal(mean=0))(x)
    predictions = keras.layers.Dense(units=3, activation="softmax",
                                     kernel_initializer=keras.initializers.RandomNormal(mean=0))(x)

    # Create model using forzen base layers and new FC layers
    model = keras.models.Model(inputs=base_model.input, 
                               outputs=predictions, 
                               name=run_id) 
    
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
    #pretrained_lr = 0.0001 
    #new_lr = 10 * pretrained_lr 
    
            # Create multioptimizer -----
            #optimizers = [keras.optimizers.Adam(learning_rate=pretrained_lr),
            #              keras.optimizers.Adam(learning_rate=new_lr)]

            # Layer objects for pre-trained and FC layers
            #block_17_layers = [ model.get_layer(name=name) for name in block_17_names ]
            #new_fc_layers = model.layers[-3:]

            # (Optimizer, layer) pairs 
            #block_17_optimizers_and_layers =  [(optimizers[0], block_17_layers)]  #[  (optimizers[0],layer) for layer in block_17_layers ]
            #new_fc_optimizers_and_layers = [(optimizers[1], new_fc_layers)]  #[  (optimizers[1],layer) for layer in new_fc_layers ]
            #optimizers_and_layers = block_17_optimizers_and_layers + new_fc_optimizers_and_layers

            # Optimizer with different learning rates across layers
            #optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    
    # LR MULTIPLIER
                #multipliers = {}
                #print(block_17_names)
                #optimizer = LRMultiplier('adam', multipliers)
    
    
    # Standard Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    #optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    #optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
    
    # ---------------------------
    
    
    # LOSS FUNCTION AND METRICS
    # -------------------------------------
    # Apply label smoothing factor, default is 0 (no smoothing)
    loss_func = keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth_factor)
        
    metrics_list = ['accuracy',
                    keras.metrics.AUC( multi_label=True )] 
    
    
    # COMPILE 
    # -------------------------------------
    model.compile(optimizer=optimizer ,
                loss=loss_func ,
                metrics=metrics_list)
    
    return model


# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

# HOSSEINZADEH
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

def ResNet50_Hosseinzadeh(run_id, label_smoothing):
    
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
                               name=run_id) 


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




def ResNet50(run_id, label_smooth_factor=0, img_width=224, img_height=224):
    """
    Creates a Keras model and from a base pre-trained model and newly defined output layers.
    Compiles the model with defined optimizer, loss and metrics.
    
    Returns: compiled Keras model ready for training
    """
        
    # DEFINING MODEL LAYERS
    # ---------------------------
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights="imagenet",
                                                      input_shape=(img_width,img_height,3))
    
    
    base_model.trainable = False 
    '''
    # Define output layers 
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(units=64, activation="relu")(x)
    predictions = keras.layers.Dense(units=3, activation="softmax")(x)

    # Create model using forzen base layers and new FC layers
    model = keras.models.Model(inputs=base_model.input, 
                               outputs=predictions, 
                               name=run_id) 
    '''
    inputs = keras.Input(shape=(img_width,img_height,3))
    x = keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(units=64, activation="relu")(x)
    outputs = keras.layers.Dense(units=3, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, 
                               outputs=outputs, 
                               name=run_id) 

    # OPTIMIZERS
    # -------------------------------------
    # Standard Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    #optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    #optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
    # ---------------------------
    
    
    # LOSS FUNCTION AND METRICS
    # -------------------------------------
    # Apply label smoothing factor, default is 0 (no smoothing)
    loss_func = keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth_factor)
        
    metrics_list = ['accuracy',
                    keras.metrics.AUC( multi_label=True )] 
    
    # COMPILE 
    # -------------------------------------
    model.compile(optimizer=optimizer ,
                loss=loss_func ,
                metrics=metrics_list,
                run_eagerly=True)
    
    return model


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

def create_lr_scheduler_cb():
    """
    Callback function to drop LR by factor of 10 at
    the 5th and 10th epoch
    """
    def scheduler(epoch, learning_rate):
        if epoch == 5 or epoch == 10:
            return learning_rate * 0.1
        else:
            return learning_rate
        
    cb = keras.callbacks.LearningRateScheduler(schedule=scheduler, verbose=1)
    
    return cb

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

def save_training_plots(history, model_name, num_epochs):
    # The following code will save an image showing the above metrics for the model during the training process. 

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, num_epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, num_epochs), history.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'./output/results/{model_name}/train_loss.png')
    
    plt.figure()
    plt.plot(np.arange(0, num_epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, num_epochs), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f'./output/results/{model_name}/train_accuracy.png')


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
    lr_schedule_cb = create_lr_scheduler_cb()
    cb_list = [checkpoint_cb, lr_schedule_cb]
    
        
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


# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    model = ResNet50_Mahbod("Test")
    #for layer in model.layers:
    #    if layer.trainable:
    #        print(layer.name
    

    
    
  
    
        
