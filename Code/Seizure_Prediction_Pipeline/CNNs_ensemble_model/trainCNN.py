from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
import tensorflow as tf
from keras.utils import to_categorical
import os
import numpy as np

def train_CNN(training_data_i,training_labels,validation_data,validation_labels):
    # model training
    swish_function = tf.keras.activations.swish
    nr_filters = 32
    filter_size = 5
    
    input_layer = Input(shape=(1280,19,1))
    
    x = Conv2D(nr_filters,(filter_size,3),(1,1),'same')(input_layer)
    x = Conv2D(nr_filters,(filter_size,3),(2,2),'same')(x)
    x = SpatialDropout2D(0.5)(x)
    x = Activation(swish_function)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(nr_filters*2,(filter_size,3),(1,1),'same')(x)
    x = Conv2D(nr_filters*2,(filter_size,3),(2,2),'same')(x)
    x = SpatialDropout2D(0.5)(x)
    x = Activation(swish_function)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(nr_filters*4,(filter_size,3),(1,1),'same')(x)
    x = Conv2D(nr_filters*4,(filter_size,3),(2,2),'same')(x)
    x = SpatialDropout2D(0.5)(x)
    x = Activation(swish_function)(x)
    x = BatchNormalization()(x)
    
    x = GlobalAveragePooling2D()(x)
    
    x = Dropout(0.5)(x)
    
    x = Dense(2)(x)
    
    output_layer = Activation('softmax')(x)
    
    model = Model(input_layer,output_layer)
    model.compile(optimizer=Adam(learning_rate=3e-4), loss='binary_crossentropy',metrics='acc')
    model.summary()
            
           
            
#            if os.path.isdir(f'{root_path}/Results_mauro/Patient {patient}/')==False:
#                os.mkdir(f'{root_path}/Results_mauro/Patient {patient}/')
#            
#            model_checkpoint_cb = ModelCheckpoint(f'{root_path}/Results_mauro/Patient {patient}/seizure_prediction_model_0.h5', 'val_loss',
#                                                      save_best_only = True, verbose = 1, mode = 'min')
            
    early_stopping_cb = EarlyStopping(monitor = 'val_loss', patience = 50)

#            callbacks_parameters = [model_checkpoint_cb, early_stopping_cb]
            
    callbacks_parameters = [early_stopping_cb]
    training_labels = to_categorical(training_labels,2)
    validation_labels = to_categorical(validation_labels,2)
    
    norm_values = [np.mean(training_data_i),np.std(training_data_i)]
    training_data_i = (training_data_i-norm_values[0])/norm_values[1]
    validation_data = (validation_data-norm_values[0])/norm_values[1]
    
    model.fit(training_data_i, training_labels, epochs = 500,
              verbose = 1, validation_data = (validation_data, validation_labels),
              callbacks = callbacks_parameters)
    
    return [model, validation_data, validation_labels, norm_values]
        
        
        
def train_CNN_and_save(training_data_i,training_labels,validation_data,validation_labels,nn,patient):
    # model training - Fábio
    swish_function = tf.keras.activations.swish
    nr_filters = 32
    filter_size = 5
    
    input_layer = Input(shape=(1280,19,1))
    
    x = Conv2D(nr_filters,(filter_size,3),(1,1),'same')(input_layer)
    x = Conv2D(nr_filters,(filter_size,3),(2,2),'same')(x)
    x = SpatialDropout2D(0.5)(x)
    x = Activation(swish_function)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(nr_filters*2,(filter_size,3),(1,1),'same')(x)
    x = Conv2D(nr_filters*2,(filter_size,3),(2,2),'same')(x)
    x = SpatialDropout2D(0.5)(x)
    x = Activation(swish_function)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(nr_filters*4,(filter_size,3),(1,1),'same')(x)
    x = Conv2D(nr_filters*4,(filter_size,3),(2,2),'same')(x)
    x = SpatialDropout2D(0.5)(x)
    x = Activation(swish_function)(x)
    x = BatchNormalization()(x)
    
    x = GlobalAveragePooling2D()(x)
    
    x = Dropout(0.5)(x)
    
    x = Dense(2)(x)
    
    output_layer = Activation('softmax')(x)
    
    model = Model(input_layer,output_layer)
    model.compile(optimizer=Adam(learning_rate=3e-4), loss='binary_crossentropy',metrics='acc')
    model.summary()
    
   
    root_path = '/media/fabioacl/EPILEPSIAE Preprocessed Data/Cose di Mauri/'
    if os.path.isdir(f'{root_path}/Results_mauro/Patient {patient}/')==False:
        os.mkdir(f'{root_path}/Results_mauro/Patient {patient}/')
    
    model_checkpoint_cb = ModelCheckpoint(f'{root_path}/Results_mauro/Patient {patient}/seizure_prediction_model_'+str(nn)+'.h5', 'val_loss',
                                              save_best_only = True, verbose = 1, mode = 'min')
    
    early_stopping_cb = EarlyStopping(monitor = 'val_loss', patience = 50)

    callbacks_parameters = [model_checkpoint_cb, early_stopping_cb]
    
    training_labels = to_categorical(training_labels,2)
    validation_labels = to_categorical(validation_labels,2)
    
    norm_values = [np.mean(training_data_i),np.std(training_data_i)]
    training_data_i = (training_data_i-norm_values[0])/norm_values[1]
    validation_data = (validation_data-norm_values[0])/norm_values[1]
    
    model.fit(training_data_i, training_labels, epochs = 500,
              verbose = 1, validation_data = (validation_data, validation_labels),
              callbacks = callbacks_parameters)
    
    
    if os.path.isdir(f'{root_path}/Results_mauro/Patient {patient}/')==False:
        os.mkdir(f'{root_path}/Results_mauro/Patient {patient}/')
    np.save(f'{root_path}/Results_mauro/Patient {patient}/norm_values_'+str(nn)+'.npy',norm_values)
    
    
    return [model, validation_data, validation_labels, norm_values]
        
        