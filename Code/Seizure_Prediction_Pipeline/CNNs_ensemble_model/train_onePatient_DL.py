# code to train, for one patient, the CNNs ensemble.
# it also returns the best preictal period

import os
import numpy as np
import utils
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
import tensorflow as tf
from keras.utils import to_categorical

from trainCNN import train_CNN
from trainCNN import train_CNN_and_save

# due to the np.delete function
import warnings
warnings.filterwarnings("ignore")
def train_model_DeepLearning(patient):
    
    # make variables to store the grid-search performance for each preictal period
    sop_gridsearch_values=[20,30,40,50]
    # making the vector that will save the performances
    performance_values=np.zeros([len(sop_gridsearch_values)])
    
    # k runs for each pre-ictal
    k=1
    # list of n final models
    n=1
    
    # where the data is
    root_path = "/media/fabioacl/EPILEPSIAE Preprocessed Data"
    path = f"{root_path}/Doentes_mauro/"
    # go to where the data is
    os.chdir(path)
    
    os.chdir("pat_"+str(patient)+"_splitted")
        
    # load training seizures
    seizure_1_data=np.load("seizure_0_data.npy");
    seizure_2_data=np.load("seizure_1_data.npy");
    seizure_3_data=np.load("seizure_2_data.npy");
        
    # loading seizure datetimes
    seizure_1_datetime=np.load("feature_datetimes_0.npy");
    seizure_2_datetime=np.load("feature_datetimes_1.npy");
    seizure_3_datetime=np.load("feature_datetimes_2.npy");
        
    # load seizure onset's
    seizure_information=np.load("all_seizure_information.pkl",allow_pickle=True)
    seizure_onset_1=float(seizure_information[0][0])
    seizure_onset_2=float(seizure_information[1][0])
    seizure_onset_3=float(seizure_information[2][0])
    
    # removing the sph
    [seizure_1_data,seizure_1_datetime]=utils.removeSPHfromSignal(seizure_1_data,seizure_1_datetime, seizure_onset_1)
    [seizure_2_data,seizure_2_datetime]=utils.removeSPHfromSignal(seizure_2_data,seizure_2_datetime, seizure_onset_2)
    [seizure_3_data,seizure_3_datetime]=utils.removeSPHfromSignal(seizure_3_data,seizure_3_datetime, seizure_onset_3)
    
    # concatenate data
    training_data=np.concatenate([seizure_1_data, seizure_2_data, seizure_3_data], axis=0)
    
    del seizure_1_data
    del seizure_2_data
    del seizure_3_data
    
    for i in range(0,len(sop_gridsearch_values)):
        # repeat this motherfucker K times
        for kk in range(0,k):
            # get seizure labels
            seizure_1_labels=utils.getLabelsForSeizure(seizure_1_datetime,sop_gridsearch_values[i],seizure_onset_1)
            seizure_2_labels=utils.getLabelsForSeizure(seizure_2_datetime,sop_gridsearch_values[i],seizure_onset_2)
            seizure_3_labels=utils.getLabelsForSeizure(seizure_3_datetime,sop_gridsearch_values[i],seizure_onset_3)
            
            # concatenate labels
            training_labels=np.concatenate([seizure_1_labels, seizure_2_labels, seizure_3_labels], axis=0)
        
            training_data_i,validation_data,training_labels,validation_labels = train_test_split(training_data, training_labels,
                                                                                       test_size = 0.2, random_state = 0,
                                                                                       shuffle=True, stratify = training_labels)    
        
            # Data Balancing - Sampling
            # we will use systematic sampling (random undersampling)
            idx_selected = utils.systematic_random_undersampling(training_labels)
            training_data_i=training_data_i[idx_selected,:]
            training_labels=training_labels[idx_selected]
            
            # reshape the data
            training_data_i=np.expand_dims(training_data_i,axis=3)
            validation_data=np.expand_dims(validation_data,axis=3)
        
            [model, validation_data, validation_labels, norm_values]=train_CNN(training_data_i,training_labels,
                                                                               validation_data,validation_labels)
            
            ###################### Performance Evaluation #########################
            
            ###################### Classification #######################
            # predicted_labels=[]
            # for i in range(0,len(validation_labels)):
            #     current_testing_data = (validation_data[i]-norm_values[0])/norm_values[1]
            #     current_test
            #     y_pred = model.predict(current_testing_data)
            #     y_pred = np.argmax(y_pred,axis=1)
            #     predicted_labels.append(y_pred)
            predicted_labels = model.predict(validation_data)
            predicted_labels = np.argmax(predicted_labels, axis=1)
            validation_labels = np.argmax(validation_labels, axis=1)
            
                
            tn, fp, fn, tp = confusion_matrix(validation_labels, predicted_labels).ravel()    
                        
            performance=np.sqrt(utils.specificity(tn,fp,fn,tp)*utils.sensitivity(tn,fp,fn,tp))
                        
            # calcular performance pre_ictal com todos os k
            performance_values[i]=performance_values[i]+performance
    
    
    index_best_models=np.argmax(performance_values)
    best_sop=sop_gridsearch_values[index_best_models]
    
    np.save(f'where do you want your models to be saved',performance_values)
    
    # training the final models
    # repeat this motherfucker n times
    for nn in range(0,n):
        # get seizure labels
        seizure_1_labels=utils.getLabelsForSeizure(seizure_1_datetime,best_sop,seizure_onset_1)
        seizure_2_labels=utils.getLabelsForSeizure(seizure_2_datetime,best_sop,seizure_onset_2)
        seizure_3_labels=utils.getLabelsForSeizure(seizure_3_datetime,best_sop,seizure_onset_3)
        
        # concatenate labels
        training_labels=np.concatenate([seizure_1_labels, seizure_2_labels, seizure_3_labels], axis=0)
        
        training_data,validation_data,training_labels,validation_labels = train_test_split(training_data, training_labels,
                                                                                       test_size = 0.2, random_state = 0,
                                                                                       shuffle=True, stratify = training_labels)    
        
        # Data Balancing - Sampling
        # we will use systematic sampling (random undersampling)
        idx_selected = utils.systematic_random_undersampling(training_labels)
        training_data_i=training_data[idx_selected,:]
        training_labels=training_labels[idx_selected]
        
        # reshape the data
        training_data_i=np.expand_dims(training_data_i,axis=3)
        validation_data=np.expand_dims(validation_data,axis=3)
    
        train_CNN_and_save(training_data_i,training_labels,validation_data,validation_labels,nn,patient)
            
    # we return the set of models that belong to a given preictal
    return best_sop

