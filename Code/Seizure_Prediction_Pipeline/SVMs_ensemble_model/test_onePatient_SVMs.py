# test code for one patient, for the SVMs

import os
import numpy as np
import utils

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# due to the np.delete function
import warnings
warnings.filterwarnings("ignore")

def testOnePatient(patient,sop, number_features,total_seizures,c_value):
       
    channels=np.load("channel_names.pkl",allow_pickle=True)
    channels=channels[0:19]
    
    linear_feature_names=["Delta_power","Theta_power","Alpha_power","Beta_power",
        "Gamma1_power","Gamma2_power","Gamma3_power","Gamma4_power",
        "Relative_delta_power","Relative_theta_power","Relative_alpha_power",
        "Relative_beta_power","Relative_gamma1_power","Relative_gamma2_power",
        "Relative_gamma3_power","Relative_gamma4_power","Total_power",
        "Alpha_peak_frequency","Mean_frequency","Ratio_delta_theta",
        "Ratio_delta_alpha","Ratio_delta_beta","Ratio_delta_gamma1",
        "Ratio_delta_gamma2","Ratio_delta_gamma3","Ratio_theta_alpha",
        "Ratio_theta_beta","Ratio_theta_gamma1","Ratio_theta_gamma2",
        "Ratio_theta_gamma3","Ratio_alpha_beta","Ratio_alpha_gamma1",
        "Ratio_alpha_gamma2","Ratio_alpha_gamma3","Ratio_beta_gamma1",
        "Ratio_beta_gamma2","Ratio_beta_gamma3","Ratio_gamma1_gamma2",
        "Ratio_gamma1_gamma3","Ratio_gamma2_gamma3",
        "Ratio_beta_over_alpha_theta","Ratio_theta_over_alpha_beta",
        "Normalized_mean_intensity","Mean_intensity","Std","Kurtosis",
        "Skewness","Activity","Mobility","Complexity","Spectral_edge_frequency",
        "Spectral_edge_power","Decorrelation_time","energy_D1","energy_D2",
        "energy_D3","energy_D4","energy_D5","energy_A5"]
    
    features_names=[]
    for i in range (0,len(channels)):
        for j in range(0,len(linear_feature_names)):
            features_names.append(channels[i]+"_"+linear_feature_names[j])
    features_names=np.array(features_names)
    
     # where the data is
    os.chdir("..")
    os.chdir("..")
    os.chdir("..")
    os.chdir("Data\Features_data")

    sph=10
    window_length=5
    
    os.chdir("pat_"+str(patient)+"_features")
    
    # load training seizures
    seizure_1_data=np.load("pat_"+str(patient)+"_seizure_0_features.npy");
    seizure_2_data=np.load("pat_"+str(patient)+"_seizure_1_features.npy");
    seizure_3_data=np.load("pat_"+str(patient)+"_seizure_2_features.npy");
        
    # loading seizure datetimes
    seizure_1_datetime=np.load("feature_datetimes_0.npy");
    seizure_2_datetime=np.load("feature_datetimes_1.npy");
    seizure_3_datetime=np.load("feature_datetimes_2.npy");
        
    # load seizure onset's
    seizure_information=np.load("all_seizure_information.pkl",allow_pickle=True)
    seizure_onset_1=seizure_information[0][0]
    seizure_onset_2=seizure_information[1][0]
    seizure_onset_3=seizure_information[2][0]
        
    # removing the sph
    [seizure_1_data,seizure_1_datetime]=utils.removeSPHfromSignal(seizure_1_data,seizure_1_datetime, seizure_onset_1)
    [seizure_2_data,seizure_2_datetime]=utils.removeSPHfromSignal(seizure_2_data,seizure_2_datetime, seizure_onset_2)
    [seizure_3_data,seizure_3_datetime]=utils.removeSPHfromSignal(seizure_3_data,seizure_3_datetime, seizure_onset_3)
        
    seizure_1_labels=utils.getLabelsForSeizure(seizure_1_datetime,sop,seizure_onset_1)
    seizure_2_labels=utils.getLabelsForSeizure(seizure_2_datetime,sop,seizure_onset_2)
    seizure_3_labels=utils.getLabelsForSeizure(seizure_3_datetime,sop,seizure_onset_3)
    
    # concatenate both validation features and labels
    training_features=np.concatenate([seizure_1_data, seizure_2_data, seizure_3_data], axis=0)
    training_labels=np.concatenate([seizure_1_labels, seizure_2_labels, seizure_3_labels], axis=0)
                        
    # reshape the training feature vector
    training_features=np.reshape(training_features,(training_features.shape[0],
                                                    training_features.shape[1]*training_features.shape[2]))
    
    del seizure_1_data
    del seizure_2_data
    del seizure_3_data
    
    del seizure_1_labels
    del seizure_2_labels
    del seizure_3_labels
    
    del seizure_1_datetime
    del seizure_2_datetime
    del seizure_3_datetime
    
    del seizure_onset_1
    del seizure_onset_2
    del seizure_onset_3
    
    ####################### Loading Testing Seizures #############################
        
    testing_features=[]
    testing_labels=[]
    testing_datetimes=[]
    testing_onsets=[]
    for seizure_k in range(3,total_seizures):
        # load training seizures
        seizure_features=np.load("pat_"+str(patient)+"_seizure_"+str(seizure_k)+"_features.npy");
        seizure_datetime=np.load("feature_datetimes_"+str(seizure_k)+".npy");
        seizure_onset=seizure_information[seizure_k][0]
        
        # removing the SPH
        [seizure_features,seizure_datetime]=utils.removeSPHfromSignal(seizure_features,seizure_datetime, seizure_onset)
        
        # retrieving the labels for one seizure
        seizure_labels=utils.getLabelsForSeizure(seizure_datetime,sop,seizure_onset)
            
        # reshape the feature vector
        seizure_features=np.reshape(seizure_features,(seizure_features.shape[0],
                                                    seizure_features.shape[1]*seizure_features.shape[2]))
       
        # we fix the label vectors
        seizure_labels=np.transpose(seizure_labels)
        
        testing_features.append(seizure_features)
        testing_labels.append(seizure_labels)
        testing_datetimes.append(seizure_datetime)
        testing_onsets.append(seizure_onset)
    
    # we transpose the feature vector to have sample x feature    
    # confirmar como está isto, tanto para features como labels
                            
        
    ################### Missing value imputation ##################################    
    # find missing values for training
    missing_values_indexes=np.unique(np.argwhere(np.isnan(training_features))[:,0])       
    training_features=np.delete(training_features,missing_values_indexes,axis=0)
    training_labels=np.delete(training_labels,missing_values_indexes,axis=0)
    
    # find missing values for testing
    for i in range(0,len(testing_labels)):
        missing_values_indexes=np.unique(np.argwhere(np.isnan(testing_features[i]))[:,0])
        
        testing_features[i]=np.delete(testing_features[i],missing_values_indexes,axis=0)
        testing_labels[i]=np.delete(testing_labels[i],missing_values_indexes,axis=0)
        testing_datetimes[i]=np.delete(testing_datetimes[i],missing_values_indexes,axis=0)
                                           
    ################## Removing Constant and Redundant Values ###########
    # remove constant features from training features
    [constant_indexes,training_features]=utils.removeConstantFeatures(training_features);
    features_names=np.delete(features_names,constant_indexes)
    
    # remove the same features from testing features
    for i in range(0,len(testing_labels)):
        testing_features[i]=np.delete(testing_features[i],constant_indexes,axis=1)
    
    #################### Standardization #######################
    # training features
    scaler = StandardScaler().fit(training_features)
    training_features=scaler.transform(training_features)
    
    for i in range(0,len(testing_labels)): 
        # testing features
        testing_features[i]=scaler.transform(testing_features[i])
    
    features_names_each_classifier=[]
    predicted_labels_each_classifier=[]
    #train 15 classifiers
    for iii in range(0,15):
        
        testing_features_j=testing_features.copy()
        
        #################### Data Balancing - Sampling ###########################
        # we will use systematic sampling (random undersampling)
        idx_selected = utils.systematic_random_undersampling(training_labels)
        training_features_i=training_features[idx_selected,:]
        training_labels_i=training_labels[idx_selected]
                                
        #################### Feature Selection #######################
        n_features=number_features
        # making the random forest feature selection
        rf = RandomForestClassifier(max_depth=10, random_state=42, n_estimators = 100).fit(training_features_i, training_labels_i)
        # sorting the features by descendent order
        feature_importance_indexes=np.argsort(((-1)*rf.feature_importances_))
        # selecting the N best features
        selected_features_indexes=feature_importance_indexes[0:n_features]
                                
        # keeping the selected features
        training_features_i=training_features_i[:,selected_features_indexes]
           
        for i in range(0,len(testing_labels)):             
            # apply feature selection to testing_labels
            testing_features_j[i]=testing_features_j[i][:,selected_features_indexes]
         
        # saving the features names
        features_names_i=features_names.copy()
        features_names_i=features_names[selected_features_indexes]
        features_names_each_classifier.append(features_names_i)
        
        #################### Classification ###########################
        predicted_labels=[]
        # Define svm model
        svm_model = svm.LinearSVC(C=c_value, dual = False)
        # Appy fit
        svm_model.fit(training_features_i, training_labels_i)
        
        for i in range(0,len(testing_labels)): 
            predicted_labels.append(svm_model.predict(testing_features_j[i]))
        
        predicted_labels_each_classifier.append(predicted_labels)
    
    # making the voting system
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
            
