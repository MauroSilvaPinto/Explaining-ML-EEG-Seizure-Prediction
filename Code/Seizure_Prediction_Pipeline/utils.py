# code with several utility functions

import datetime
from datetime import timedelta
import numpy as np
from sklearn.utils import class_weight
from scipy import stats
from sklearn import metrics

    
def convertIntoDatetime(date):
    return datetime.datetime.fromtimestamp(date)

# remove the SPH minutes from seizures
# in this case, we use a 10 minute SPH
def removeSPHfromSignal(seizure_data,seizure_datetime, seizure_onset):
    seizure_onset=convertIntoDatetime(seizure_onset)
    sph_datetime=seizure_onset-timedelta(minutes=10)
    
    final_index=0
    # we search for all datetimes and compare it with the onset-SPH time
    for i in range(len(seizure_datetime)-1,1,-1):
        current_datetime=datetime.datetime.fromtimestamp(seizure_datetime[i])
        # when we reach the onset-SPH time, we stop
        if not(current_datetime>sph_datetime):
            final_index=i
            break
    # we keep the elements from the beginning until the SPH moment
    seizure_datetime=seizure_datetime[0:final_index]
    seizure_data=seizure_data[0:final_index,:,:]
    
    # we return the data, and the datetimes
    return seizure_data, seizure_datetime
    

# remove the SPH minutes from seizures
# in this case, we use a 10 minute SPH
def getLabelsForSeizure(seizure_datetime, sop, seizure_onset):
    seizure_onset=seizure_onset=convertIntoDatetime(seizure_onset)
    preictal_datetime=seizure_onset-timedelta(minutes=(10+sop))
    
    final_index=0
    # we search for all datetimes and compare it with the onset-SPH time
    for i in range(len(seizure_datetime)-1,1,-1):
        current_datetime=datetime.datetime.fromtimestamp(seizure_datetime[i])
        # when we reach the pre-ictal onset, we stop
        if not(current_datetime>preictal_datetime):
            final_index=i
            break
    # we create our label vector, with only 0's
    labels=np.zeros(len(seizure_datetime))
    # from the pre-ictal onset to the end, we fill it with 1's
    labels[final_index:]=1
    
    return labels

# a code to remove features with var<1e-9
def removeConstantFeatures(features):
    constant_features_index=[]
    
    #finding features with corr<1e-9
    for i in range(0,features.shape[1]):
        if np.var(features[:,i])<1e-9:
            constant_features_index.append(i)
      
    # deleting these features          
    features=np.delete(features,constant_features_index,axis=1)
    return [constant_features_index,features]

# a code to remove features with corr>0.95
def removeRedundantFeatures(features):
    redundant_features_index=[]
    
    #finding features with corr>0.95
    for i in range(0,features.shape[1]):
        for j in range(i,features.shape[1]):
            if i!=j and abs(np.corrcoef(features[:,i],features[:,j])[0][1])>0.95:
                redundant_features_index.append(j)
    
    # deleting these features          
    features=np.delete(features,redundant_features_index,axis=1)
    return [redundant_features_index,features]


# a code tu compute each sample weight according to its class
    # samples from the less representative class will have a higher weight
    # samples from the more representative class will have a smaller weight
def computeSampleWeights(labels,class_weights):
        sample_weights=np.zeros(len(labels))
        
        sample_weights[np.where(labels==0)[0]]=class_weights[0]
        sample_weights[np.where(labels==1)[0]]=class_weights[1]
       
        return sample_weights
   
    
    
# a code to compute each label weight, according to its representation size
def computeBalancedClassWeights(labels):
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(labels),
                                                 y=labels)
    return class_weights
     
def specificity(tn,fp,fn,tp):
    return (tn/(tn+fp))

def sensitivity(tn,fp,fn,tp):
    return (tp/(tp+fn))


# a class to perform systematic
# random undersampling
def systematic_random_undersampling(target):
    
    # Define majority & minority classes (class with more samples vs. class with less samples)
    idx_class0 = np.where(target==0)[0]
    idx_class1 = np.where(target==1)[0]
    if len(idx_class1)>=len(idx_class0):
        idx_majority_class = idx_class1
        idx_minority_class = idx_class0
    elif len(idx_class1)<len(idx_class0):
        idx_majority_class = idx_class0
        idx_minority_class = idx_class1
    
    # Define number of samples of each group
    n_groups = len(idx_minority_class)
    n_samples = len(idx_majority_class)
    min_samples = n_samples//n_groups
    remaining_samples = n_samples%n_groups
    n_samples_per_group = [min_samples+1]*remaining_samples + [min_samples]*(n_groups-remaining_samples)
    
    # Select one sample from each group of the majority class
    idx_selected = []
    begin_idx = 0
    for i in n_samples_per_group:
        end_idx = begin_idx + i
        
        idx_group = idx_majority_class[begin_idx:end_idx]
        idx = np.random.choice(idx_group)
        idx_selected.append(idx)

        begin_idx = end_idx
        
    # Add samples from the minority class
    [idx_selected.append(idx) for idx in idx_minority_class]

    # Sort selected indexes to keep samples order
    idx_selected = np.sort(idx_selected)
    
    return idx_selected


# a code that processes the classifier output with the firing power
# and refractory behavior
def FiringPowerAndRefractoryPeriod(predicted_labels,datetimes,sop,sph,window_length):
    predicted_labels=FiringPower(predicted_labels,sop,window_length)
    predicted_labels=RefractoryPeriod(predicted_labels,datetimes,sop,sph, window_length)
    
    return predicted_labels

#a code to implement the firing power
# which is a moving average filter (low-pass)
# with the size of the pre-ictal period
def FiringPower(predicted_labels,sop,window_length):
    kernel_size =int(sop*(60/window_length))
    kernel = np.ones(kernel_size) / kernel_size
    predicted_labels = np.convolve(predicted_labels, kernel, mode='same')
    
    threshold=0.7
    predicted_labels = [1 if predicted_labels_ > threshold else 0 for predicted_labels_ in predicted_labels]
    
    return predicted_labels

# a code to implement the refractory period
# the period in which you cannot let your classifier
# send an alarm, as it has already previously sent one
def RefractoryPeriod(predicted_labels,datetimes,sop,sph,window_length):
    refractory_on=False
    for i in range(0,len(predicted_labels)):
        if refractory_on==False:
            # when a new alarm is found, a refractory period begins
            if predicted_labels[i]==1:
                refractory_on=True
                onset_alarm=convertIntoDatetime(datetimes[i])
        else:
            # if we are on the refractory period, we set the labels to 0
            # and count the refractory period
            predicted_labels[i]=0
            
            #when the refractory period reaches its period, it ends
            # and the refractory bar is set to 0
            if (convertIntoDatetime(datetimes[i]) >
                (onset_alarm+timedelta(minutes=sop+sph))):
                refractory_on=False
    
    return predicted_labels


# a code to verify if a model predicted the seizure
def didItPredictTheSeizure(predicted,labels):
    pre_ictal_length=len(np.argwhere(labels))
    pre_ictal_beginning=np.argwhere(labels)[0][0]
    did_it_predict_the_seizure=1 in predicted[pre_ictal_beginning:pre_ictal_beginning+pre_ictal_length]
    
    return did_it_predict_the_seizure


# code to retrieve the duration of time in which is possible to trigger an alarm
def calculateFPRdenominator(predicted,labels,pre_ictal,datetime, onset, window_length):
    pre_ictal_beginning=np.argwhere(labels)[0][0]
    FPR_denominator=0;
    number_of_false_alarms=calculateNumberFalseAlarms(predicted,labels)
    false_alarm_indexes=np.argwhere(predicted[0:pre_ictal_beginning])
    
    onset=convertIntoDatetime(onset)
    preictal_onset=onset-timedelta(minutes=pre_ictal)
    for i in range(0,number_of_false_alarms):
        time_false_alarm=convertIntoDatetime(datetime[false_alarm_indexes[i][0]])
        distance_alarm_preictal=preictal_onset-time_false_alarm
        if (distance_alarm_preictal.total_seconds()>pre_ictal*60):
            FPR_denominator=FPR_denominator+pre_ictal/60
        else:
            FPR_denominator=FPR_denominator+(distance_alarm_preictal.total_seconds()/3600)
    
    inter_ictal_length=(len(labels)-sum(labels))*window_length/3600       
    return inter_ictal_length-FPR_denominator


# a code to calculate the number of false alarms    
def calculateNumberFalseAlarms(predicted,labels):
     pre_ictal_beginning=np.argwhere(labels)[0][0]
     number_false_alarms=np.sum(predicted[0:pre_ictal_beginning])
    
     return number_false_alarms   

def surrogateSensitivity(predicted,labels,datetimes,onset, sop, sph):
    seizure_sensitivity=0
    surrogate_labels=shuffle_labels(datetimes, onset, sop, sph)
    seizure_sensitivity=seizure_sensitivity+didItPredictTheSeizure(predicted,surrogate_labels)
    
    return seizure_sensitivity


# code to shuffle the pre-seizure labels for the surrogate
def shuffle_labels(datetimes, onset, sop, sph):
    firing_power_threshold=0.7
    surrogate_labels=np.zeros(len(datetimes))
    
    pre_ictal_beginning=convertIntoDatetime(onset)
    pre_ictal_beginning=pre_ictal_beginning-timedelta(minutes=sop+sph)
    
    minimum_index=sop*firing_power_threshold
    maximum_index=0
    for i in range(len(datetimes)-1,0,-1):
        current_moment=convertIntoDatetime(datetimes[i])
        if pre_ictal_beginning>current_moment:
            maximum_index=i
            break
        
    sop_beginning_index=np.random.randint(minimum_index,maximum_index)
    
    end_time=convertIntoDatetime(datetimes[sop_beginning_index])
    end_time=end_time+timedelta(minutes=sop)
    
    for i in range (sop_beginning_index,len(datetimes)):
        current_moment=convertIntoDatetime(datetimes[i])
        if (current_moment>end_time):
            break
        else:
            surrogate_labels[i]=1
        
    return surrogate_labels


def t_test_one_independent_mean(population_mean, population_std, sample_mean, number_samples):   
    tt=abs(population_mean-sample_mean)/(population_std/np.sqrt(number_samples))
    
    pval = stats.t.sf(np.abs(tt), number_samples-1)*2  # two-sided pvalue = Prob(abs(t)>tt)
    #print('t-statistic = %6.3f pvalue = %6.4f' % (tt, pval))
    
    return [tt,pval]
