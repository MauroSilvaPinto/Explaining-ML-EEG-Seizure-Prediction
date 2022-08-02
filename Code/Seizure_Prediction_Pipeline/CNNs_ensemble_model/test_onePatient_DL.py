# test code for one patient for the CNNs

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Run on CPU
import numpy as np
import utils

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
import tensorflow as tf
from keras.utils import to_categorical

import pickle


# due to the np.delete function
import warnings
warnings.filterwarnings("ignore")

def testOnePatient(patient,sop,total_seizures):
   
    #number of saved CNN's
    n=1
        
    # where the data is
    root_path = "where the data is"
    path = f"where the data is"

    os.chdir(path)
    
    os.chdir("pat_"+str(patient)+"_splitted")
    
    # load seizure onset's
    seizure_information=np.load("all_seizure_information.pkl",allow_pickle=True)
        
    sph=10
    window_length=5
        
    # os.chdir("pat_"+str(patient)+"_features")
    
    ####################### Loading Testing Seizures #############################
            
    testing_data=[]
    testing_labels=[]
    testing_datetimes=[]
    testing_onsets=[]
    
    for seizure_k in range(3,total_seizures):
        # load training seizures
        seizure_data=np.load("seizure_"+str(seizure_k)+"_data.npy");
        seizure_datetime=np.load("feature_datetimes_"+str(seizure_k)+".npy");
        seizure_onset=float(seizure_information[seizure_k][0])
            
        # removing the SPH
        [seizure_data,seizure_datetime]=utils.removeSPHfromSignal(seizure_data,seizure_datetime, seizure_onset)
            
        # retrieving the labels for one seizure
        seizure_labels=utils.getLabelsForSeizure(seizure_datetime,sop,seizure_onset)
                      
        # we fix the label vectors
        seizure_labels=np.transpose(seizure_labels)
        
        # reshape the data
        seizure_data=np.expand_dims(seizure_data,axis=3)
            
        testing_data.append(seizure_data)
        testing_labels.append(seizure_labels)
        testing_datetimes.append(seizure_datetime)
        testing_onsets.append(seizure_onset)
    
    
    predicted_labels_each_classifier=[]
    #load n classifiers
    for nn in range(0,n):
        
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
        
        model.load_weights(f'where your model is ... _ Patient {patient}/seizure_prediction_model_'+str(nn)+'.h5')
        
        norm_values = np.load(f'{where your model is ... Patient {patient}/norm_values_'+str(nn)+'.npy')
            
        #################### Classification ###########################
        predicted_labels=[]
        
        for i in range(0,len(testing_labels)):
            current_testing_data = (testing_data[i]-norm_values[0])/norm_values[1]
            y_pred = model.predict(current_testing_data)
            y_pred = np.argmax(y_pred,axis=1)
            predicted_labels.append(y_pred)
        
        predicted_labels_each_classifier.append(predicted_labels)
    
    
    ## making the voting system
    number_of_tested_seizures=len(predicted_labels_each_classifier[0])
    number_of_classifiers=len(predicted_labels_each_classifier)
    for i in range(0,number_of_tested_seizures):
        voted_labels=np.zeros(len(predicted_labels_each_classifier[0][i]))
        for j in range(0,number_of_classifiers):
            voted_labels=voted_labels+predicted_labels_each_classifier[j][i]
        
        voted_labels=voted_labels/number_of_classifiers
        voted_labels=np.where(voted_labels>0.5, 1, 0)
        
        predicted_labels[i]=voted_labels

    ###################### Firing Power + Refractory #############
    for i in range(0,len(testing_labels)):
        predicted_labels[i]=utils.FiringPowerAndRefractoryPeriod(predicted_labels[i],testing_datetimes[i],sop,sph,window_length)  
    
    ##################### Performance Metrics #####################
    seizure_sensitivity=0
    fpr_denominator=0
    number_false_alarms=0
    for i in range(0,len(testing_labels)):
        seizure_sensitivity=seizure_sensitivity+utils.didItPredictTheSeizure(predicted_labels[i],testing_labels[i]) # refazer
        number_false_alarms=number_false_alarms+utils.calculateNumberFalseAlarms(predicted_labels[i],testing_labels[i]) # refazer           
        fpr_denominator=fpr_denominator+utils.calculateFPRdenominator(predicted_labels[i],testing_labels[i],sop+sph,testing_datetimes[i],testing_onsets[i], window_length) #refazer 
        
    FPR=number_false_alarms/fpr_denominator
    seizure_sensitivity=seizure_sensitivity/len(testing_labels)
            
    print("FPR/h: "+ str(FPR))
    print("Sensitivity: "+ str(seizure_sensitivity))
        
    ###################### Statistical Validation ####################
    surrogate_sensitivity=[]
                
    for i in range(0,len(testing_labels)):
        for j in range(0,30):
            surrogate_sensitivity.append(utils.surrogateSensitivity(predicted_labels[i],testing_labels[i],testing_datetimes[i],testing_onsets[i],sop,sph))
                
    surrogate_sensitivity=surrogate_sensitivity
    print("Surrogate Sensitivity "+str(np.mean(surrogate_sensitivity))+" +/- " + str(np.std(surrogate_sensitivity)))
            
    val=0
    pval=1
    print("Does it perform above chance?")
    if (np.mean(surrogate_sensitivity)<seizure_sensitivity):
        [tt,pval]=utils.t_test_one_independent_mean(np.mean(surrogate_sensitivity), np.std(surrogate_sensitivity), seizure_sensitivity, 30)
        if pval<0.05:
            print("Yes")
            val=1
        else:
            print("No")
    else:
        print("No")
                            
    return [seizure_sensitivity, FPR, np.mean(surrogate_sensitivity), np.std(surrogate_sensitivity),pval,val]
                
