import matplotlib.pyplot as plt
import numpy as np

# a very straighforward method to perform counterfactual explanations
# in a time-series, iterate each sample
# in each sample:
#   iterate all features
#   in each feature:
#       change the feature value in small steps and save the minimum absolute variation to make the classifier change its prediction
# calculate the mean of each feature absolute change
# select the three features that need the minimum variation to change the overall prediction

# note that, when we change each feature in each sample, all the remaining features have the same value
# thus, we do not account here (in counterfactual explanations) for the brain as a complex system for simplicity reasons


def counterFactualExplanation(sample,model):
    
    # number of features in a sample
    number_of_features=len(sample)
    
    sample=sample.reshape(1,len(sample))
    
    # get the current answer
    obtained_answer=model.predict(sample)
    if obtained_answer==0:
        counter_factual_answer=1
    else:
        counter_factual_answer=0

    # change step
    step=0.01
       
    minimum_change_matrix=np.zeros((2,number_of_features))
    
    # iterate from 
    for i in range(0,number_of_features):       
        # positive increment
        for j in np.linspace(0,10,int((10/step))+1):
            new_sample=sample.copy()
            new_sample[0,i]=sample[0,i]+j
            new_prediction=model.predict(new_sample)
            
            if new_prediction==counter_factual_answer:
                minimum_change_matrix[0,i]=j
                break
            
        # negative imcrement
        for j in np.linspace(0,10,int((10/step))+1):
            new_sample=sample.copy()
            new_sample[0,i]=sample[0,i]-j
            new_prediction=model.predict(new_sample)
            
            if new_prediction==counter_factual_answer:
                minimum_change_matrix[1,i]=j
                break
    
    # when there are no possible changes, lets set the 0's to 100 (an impossible value)
    for i in range(0,minimum_change_matrix.shape[0]):
        for j in range(0,minimum_change_matrix.shape[1]):
            if minimum_change_matrix[i,j]==0:
                minimum_change_matrix[i,j]=100
    
    best_increment_of_each_feature=np.zeros(number_of_features)        
    # get the lowest increment (positive or negative)
    for i in range(0,number_of_features):
        if (minimum_change_matrix[0,i]<=minimum_change_matrix[1,i]):
            best_increment_of_each_feature[i]=minimum_change_matrix[0,i]
        else:
            best_increment_of_each_feature[i]=(-1)*minimum_change_matrix[1,i]
    
    return best_increment_of_each_feature





def analyzeCounterFactualExplanations(counter_factual_explanations, feature_names,
                                      title_graph):
    
    counter_factual_matrix=np.zeros([len(counter_factual_explanations),
                                 len(counter_factual_explanations[0])])
    
    for i in range(0,len(counter_factual_explanations)):
        for j in range(0,len(counter_factual_explanations[0])):
            if counter_factual_explanations[i][j]==100:
                counter_factual_matrix[i,j]=np.NaN
            else:
                counter_factual_matrix[i,j]=counter_factual_explanations[i][j]
        
    mean_efforts=np.nanmean(counter_factual_matrix,axis=0)   
    std_efforts=np.nanstd(counter_factual_matrix,axis=0)    
    
    
    fig, ax = plt.subplots(1, 1)
    
    ax.set_title(title_graph)
    ax.bar(np.array(feature_names[np.argsort(abs(mean_efforts))]),
           mean_efforts[np.argsort(abs(mean_efforts))],
           yerr=std_efforts[np.argsort(abs(std_efforts))])
    ax.grid()
    
    ax.set_xticklabels(np.array(feature_names[np.argsort(abs(mean_efforts))]),fontsize=8,rotation=15)
    
    return [np.array(feature_names[np.argsort(abs(mean_efforts))]), mean_efforts[np.argsort(abs(mean_efforts))], std_efforts[np.argsort(abs(std_efforts))]]

            
            
            
            
        
        