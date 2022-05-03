import tensorflow as tf
from tensorflow import keras 
import tensorflow_addons as tfa
import math 
from src.model import add_regularization

def Mahbod_ResNet50_Dropout(run_id, 
                            label_smooth_factor=0, 
                            img_width=224, 
                            img_height=224, 
                            lr=0.0001,
                            dropout_rate=0.5):
    """
    Creates a ResNet50 architecture based off the Mahbod et al 
    methodology with added dropout on the new FC layers.
    """ 
    
    # DEFINING MODEL LAYERS
    # ---------------------------
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights="imagenet",
                                                      input_shape=(img_width,img_height,3))
    
    base_model.trainable = False # Blocks 1-17 Frozen as in Mahbod et al.
    
    
    # Define output layers (Mahbod et al. used here)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(rate=dropout_rate)(x)
    x = keras.layers.Dense(units=64, 
                           activation="relu", 
                           kernel_initializer=keras.initializers.RandomNormal(mean=0))(x)
    x = keras.layers.Dropout(rate=dropout_rate)(x)
    predictions = keras.layers.Dense(units=3, 
                           activation="softmax", 
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
    
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    #optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    #optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
    
    
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

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

def Mahbod_Resnet50_CosineLRDecay(run_id, 
                                label_smooth_factor=0, 
                                img_width=128, 
                                img_height=128, 
                                lr=0.0001,
                                dropout_rate=0,
                                train_size=0,
                                batch_size=64,
                                num_epochs=15):
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
    #
    x = keras.layers.Dense(units=64, 
                           activation="relu", 
                           kernel_initializer=keras.initializers.RandomNormal(mean=0))(x)
    #x = keras.layers.Dropout(rate=dropout_rate)(x)
    predictions = keras.layers.Dense(units=3, 
                           activation="softmax", 
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
    
    # OPTIMIZER
    # -------------------------------------  
    
    # decay steps = (batches per epoch) * (number of epochs)
    steps = (train_size // batch_size) * (num_epochs)
    
    # Cosine learning rate decay 
    lr_decay_function = keras.experimental.CosineDecay(initial_learning_rate=lr,
                                                        decay_steps=steps,
                                                        alpha=lr*0.01) # minimum learning rate
    
     
    # Standard Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=lr_decay_function)
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


def Hosseinzadeh_ResNet50_CosineLRDecay(run_id="Hoss", 
                                        label_smooth_factor=0,
                                        img_width=225, 
                                        img_height=300, 
                                        lr=(math.e)**(-5), 
                                        dropout_rate=0.5,
                                        weight_decay=(math.e)**(-5),
                                        min_lr_factor=0.01,
                                        train_size=0,
                                        batch_size=32,
                                        num_epochs=15):
    
    # DEFINING MODEL LAYERS
    # ---------------------------
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights="imagenet",
                                                      input_shape=(img_width,img_height,3))
    
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

    
    # Add L2 regularization to all layers - Source: https://sthalles.github.io/keras-regularizer/
    regularizer = tf.keras.regularizers.l2(l=0.0001)
    model = add_regularization(model=model, regularizer=regularizer)
    
    # OPTIMIZERS
    # -------------------------------------  
    
    # decay steps = (batches per epoch) * (number of epochs)
    steps = (train_size // batch_size) * (num_epochs)
     
    # Cosine learning rate decay 
    lr_decay_function = keras.experimental.CosineDecay(initial_learning_rate=lr,
                                                        decay_steps=steps,
                                                        alpha=lr*min_lr_factor) # minimum learning rate
    
    optimizer = tfa.optimizers.AdamW(weight_decay=weight_decay,
                                     learning_rate=lr_decay_function,
                                     beta_1=0.9,
                                     beta_2=0.999,
                                     epsilon=(math.e)**(-8))
    
    # LOSS FUNCTION AND METRICS
    # -------------------------------------
    loss_func = keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth_factor)
    metrics_list = ['accuracy',
                    keras.metrics.AUC( multi_label=True )] 
    
    
    # COMPILE 
    # -------------------------------------
    model.compile(optimizer=optimizer ,
                loss=loss_func ,
                metrics=metrics_list)
                

    return model

    
#if __name__ == '__main__':
    #model = Mahbod_ResNet50_Dropout(run_id="TEST")
    #model = Mahbod_Resnet50_CosineLRDecay("TEST")
    #print(model.summary())
    
            