import tensorflow as tf
from tensorflow import keras 

def Mahbod_ResNet50_Dropout(run_id, label_smooth_factor=0):
    """
    Creates a ResNet50 architecture based off the Mahbod et al 
    methodology with added dropout on the new FC layers.
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
    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(units=64, activation="relu")(x)
    x = keras.layers.Dropout(rate=0.5)(x)
    predictions = keras.layers.Dense(units=3, activation="softmax")(x)

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
        
    # ADD WEIGHT DECAY
    # -------------------------------------
    # source: https://jricheimer.github.io/keras/2019/02/06/keras-hack-1/
    alpha = 0.00002  # weight decay coefficient
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
            layer.add_loss(keras.regularizers.l2(alpha)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(keras.regularizers.l2(alpha)(layer.bias))
    
    
    # OPTIMIZERS
    # -------------------------------------
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
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


    
if __name__ == '__main__':
    model = Mahbod_ResNet50_Dropout("TEST", label_smooth_factor=0)
    
            