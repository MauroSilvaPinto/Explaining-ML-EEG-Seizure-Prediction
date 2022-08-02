# a code that shows the constructed explanations
# this code is organised as follows:
    # 1) getting the Firing Power, datetimes, and vigilance state vectors
    # 2) get the classifier time plots
    # 3) plot Feature Influence using logreg coefficients
    # 4) get the Partial Dependence Plots
    # 5) get Beeswarm summary of Shap Values
    # 6) Calibration/Scatter Plots of all features individually
    # 7) explaining a moment of interest (scatterplots and counterfactual explanations)
        

import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime
import numpy as np
import utils

from counterFactualExplanation import analyzeCounterFactualExplanations
from counterFactualExplanation import counterFactualExplanation
from test_OnePatient_getPlots_logReg import testOnePatientGetPlots 

patient=8902
seizures=5
sop=25
k=7

#%% 1) getting the Firing Power, datetimes, and vigilance state vectors

# get the classifier time plots
[datetimes, seizure_datetime, labels, alarms,
 firing_power_output, log_reg, feature_names,
 feature_data, labels_classification, training_features,training_labels]=testOnePatientGetPlots(patient,sop,k,seizures)

# load the vigilance data obtained from the model developed in:
# https://eg.uc.pt/handle/10316/97971?locale=pt

vigilance=np.load("pat_"+str(patient)+"_vigilance",allow_pickle=True)
vigilance_datetimes=np.load("pat_"+str(patient)+"_datetimes",allow_pickle=True)

for i in range(0,len(vigilance)):
    vigilance[i]=np.abs(vigilance[i]-1)
    vigilance[i]=np.clip(vigilance[i],0.05,0.95)


#%% 2) classifier time plots

### re-arrange data for plotting
alarms_scatter=alarms.copy()
fp_scatter=firing_power_output.copy()
classification_scatter=labels_classification.copy()
pre_ictal_labels_scatter=labels.copy()
feature_names_ICE=feature_names.copy()

new_datetimes=[]
for i in range(0,seizures-3):
    new_datetimes_i=[]
    for j in range(0,len(datetimes[i])):
        new_datetimes_i.append(utils.convertIntoDatetime(datetimes[i][j]))
    new_datetimes.append(new_datetimes_i)
datetimes=new_datetimes

datetimes_new=[]
firing_power_output_new=[]
labels_new=[]
alarms_new=[]
# filling the missing data
for i in range(0,len(datetimes)):
    labels[i]=labels[i].tolist()
    firing_power_output[i]=firing_power_output[i].tolist()
    
    datetimes_new_i=[]
    firing_power_output_new_i=[]
    labels_new_i=[]
    alarms_new_i=[]    
    for kk in range(0,k):
        exec("features_data_plot_"+str(i)+"_"+str(kk)+"=[]")
    
    for j in range(0,len(datetimes[i])-1):
        time_difference=datetimes[i][j+1]-datetimes[i][j]
        time_difference=time_difference.seconds
    
        datetimes_new_i.append(datetimes[i][j])
        firing_power_output_new_i.append(firing_power_output[i][j])
        labels_new_i.append(labels[i][j])
        alarms_new_i.append(alarms[i][j])
        for kk in range(0,k):
            exec("features_data_plot_"+str(i)+"_"+str(kk)+".append(feature_data[i][j,kk])")
            
        if time_difference<=5:
            pass
        else:
            new_datetime=datetimes[i][j]+datetime.timedelta(0,5)
            while(time_difference>5):
                datetimes_new_i.append(new_datetime)
                labels_new_i.append(np.NaN)
                alarms_new_i.append(np.NaN)
                firing_power_output_new_i.append(np.NaN)
                
                time_difference=datetimes[i][j+1]-new_datetime
                time_difference=time_difference.seconds
                new_datetime=new_datetime+datetime.timedelta(0,5)
                
                for kk in range(0,k):
                    eval("features_data_plot_"+str(i)+"_"+str(kk)+".append(np.NaN)")
    
    datetimes_new.append(datetimes_new_i)
    firing_power_output_new.append(firing_power_output_new_i)
    labels_new.append(labels_new_i)
    alarms_new.append(alarms_new_i)
    

datetimes=datetimes_new
firing_power_output=firing_power_output_new
labels=labels_new
alarms=alarms_new


## plotting Firing Power output throughout time
for i in range (0,seizures-3):
    plt.figure()  

    plt.plot(datetimes[i],firing_power_output[i],'k',alpha=0.7)
    plt.plot(datetimes[i],np.linspace(0.7, 0.7, len(datetimes[i])),linestyle='--',
             color='black',alpha=0.7)
   
    plt.grid()
    
    plt.ylim(0,1)
    plt.xlim(datetimes[i][0],datetimes[i][len(datetimes[i])-1])
    
    for alarm_index in np.where(np.diff(alarms[i])==1)[0]:
        plt.plot(datetimes[i][alarm_index], firing_power_output[i][alarm_index],
                 marker='^', color='maroon',markersize=10)
   
    plt.fill_between(datetimes[i], 0.7, np.array(firing_power_output[i]), where=np.array(firing_power_output[i])>0.7,
                     facecolor='brown', alpha=0.5)
    
    plt.fill_between(datetimes[i], 0, 1, where=np.array(datetimes[i])>
                     np.array(datetimes[i][np.where(np.diff(labels[i])==1)[0][0]]),
                     facecolor='moccasin', alpha=0.5)
    
    # colour inter-ictal period
    plt.axvline(x = datetimes[i][np.where(np.diff(labels[i])==1)[0][0]], color = 'k',
                alpha = 0.7, linestyle='--',linewidth=0.8)
    
    plt.gcf().autofmt_xdate()
    xfmt = md.DateFormatter('%H:%M:%S')
    ax=plt.gca()
    ax.xaxis.set_major_formatter(xfmt)
    ax.yaxis.set_ticks([0,0.05,0.2,0.4,0.6,0.8,0.95,1.0])
    ax.yaxis.set_ticklabels(["0","sleep","0.2","0.4","0.6","0.8","awake","1.0"])
    plt.title("Patient "+str(patient)+", Seizure "+str(i+3+1))

    plt.plot(vigilance_datetimes[i], vigilance[i],alpha=0.4)

#%% 3) Plotting Feature Influence using logreg coefficients

# a 1 unit increase in X1 will increase result in b increase in the log-odds ratio of pre-ictal/inter-ictal
# by towards dta science : a simple interpretation of logistic regression coefficients
    
for i in range(0,len(feature_names)):
    feature_names[i]=feature_names[i].replace("_","\n")

fig, ax = plt.subplots(1, 1)

ax.bar(np.array(feature_names),log_reg.coef_[0])
ax.grid()
ax.set_xticklabels(np.array(feature_names),fontsize=8,rotation=15)
ax.set_title("Logistic Regression Coefficients")


#%% 4) get the Partial Dependence Plots

from sklearn.inspection import PartialDependenceDisplay
features = [0, 1,2,3,4,5,6]
PartialDependenceDisplay.from_estimator(log_reg, training_features, features, kind='both',
                                        ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
                                        pd_line_kw={"color": "tab:orange", "linestyle": "--"},
                                        feature_names=feature_names_ICE)

#%% 5) get Beeswarm summary of Shap Values
        
import shap

feature_shapley_general=feature_data[0]
for i in range(1,len(feature_data)):
    feature_shapley_general=np.concatenate((feature_shapley_general,
                                            feature_data[i]),axis=0)

explainer=shap.Explainer(log_reg,feature_shapley_general, feature_names=feature_names)
shap_values=explainer(feature_shapley_general)

fig, ax = plt.subplots(1, 1)
shap.plots.beeswarm(shap_values)
plt.title("Beeswarm summary of Shap Values")


#%% 6) Calibration/Scatter Plots of all features individually

for i in range (0,seizures-3):
    
    fig, axs=plt.subplots(k,2)
    fig.suptitle("Patient "+str(patient)+", Seizure "+str(i+3+1))
    axs[0,0].set_title('Real Labels')
    axs[0,1].set_title('Predicted Labels')
    for j in range(0,k):
        indexes_labels_1=np.where(np.array(pre_ictal_labels_scatter[i])==1)
        indexes_labels_0=np.linspace(0,len(fp_scatter[i])-1,len(fp_scatter[i]))
        indexes_labels_0=np.delete(indexes_labels_0,indexes_labels_1).astype(int)
        
        indexes_predicted_1=np.where(np.array(classification_scatter[i])==1)
        indexes_predicted_0=np.linspace(0,len(fp_scatter[i])-1,len(fp_scatter[i]))
        indexes_predicted_0=np.delete(indexes_predicted_0,indexes_predicted_1).astype(int)
        
        m,b=np.polyfit(feature_data[i][:,j],fp_scatter[i],1)
        
        axs[j,0].plot(feature_data[i][:,j], m*feature_data[i][:,j] + b, color='k')
        
        axs[j,0].scatter(feature_data[i][indexes_labels_0,j],np.array(fp_scatter[i])[indexes_labels_0],
           marker='o', alpha=0.5, c="gainsboro")    
        
        axs[j,0].scatter(feature_data[i][indexes_labels_1,j],np.array(fp_scatter[i])[indexes_labels_1],
           marker='x',alpha=0.5, c="steelblue")
        
        #axs[j,0].set(xlabel='Feature Value \n Real Labels', ylabel='FP')
        axs[j,0].set_xlabel("Feature value", fontsize=6)
        axs[j,0].set_ylabel("FP", fontsize=6)
        #axs[j,0].set_title(feature_names_ICE[j], fontsize=8)
        axs[j,0].set_yticklabels([0, 0.5, 1], fontsize=5)
        axs[j,0].set_xticklabels([-2, 0, 2, 4, 6, 8], fontsize=5)
        
        
        axs[j,0].set_ylim(0,1)
        axs[j,0].set_xlim(-2,8)
        
        if j!=k-1:
            axs[j,0].get_xaxis().set_visible(False)
        
        
        m,b=np.polyfit(feature_data[i][:,j],fp_scatter[i],1)
        
        axs[j,1].plot(feature_data[i][:,j], m*feature_data[i][:,j] + b, color='k')
        axs[j,1].scatter(feature_data[i][indexes_predicted_0,j],np.array(fp_scatter[i])[indexes_predicted_0],
           marker='o', alpha=0.5, c="gainsboro")  
        
        axs[j,1].scatter(feature_data[i][indexes_predicted_1,j],np.array(fp_scatter[i])[indexes_predicted_1],
           marker='x',alpha=0.5, c="steelblue")
        #axs[j,1].set_title(feature_names_ICE[j], fontsize=8)

       #axs[j,0].set(xlabel='Feature Value \n Real Labels', ylabel='FP')
        axs[j,1].set_xlabel("Feature value", fontsize=6)
        axs[j,1].set_ylabel("FP", fontsize=6)
        #axs[j,1].set_title(feature_names_ICE[j], fontsize=8)
        axs[j,1].set_yticklabels([0, 0.5, 1], fontsize=5)
        axs[j,1].set_xticklabels([-2, 0, 2, 4, 6, 8], fontsize=5)

        axs[j,1].set_ylim(0,1)
        axs[j,1].set_xlim(-2,8)
        if j!=k-1:
            axs[j,1].get_xaxis().set_visible(False)

#%% 7) explaining a moment of interest
            # as example, we explain the small peak at 14h from seizure 4
            # we present scatter/calibration plots and counterfactual explanations

       
# # plotting Firing Power output throughout time
# # check the samples i need to retrieve manually
#for i in range (0,seizures-3):
#     plt.figure()  
#
#     plt.plot(fp_scatter[i],'k',alpha=0.7)
#     plt.plot(np.linspace(0.7, 0.7, len(fp_scatter[i])),linestyle='--',
#             color='black',alpha=0.7)
#  
#     plt.title("Patient "+str(patient)+", Seizure "+str(i+3+1))


# seizure 4
# how to explain the 1st peak, at 14h00, that went until FP=0.31?
    #from samples 4291 to 4558
selected_samples=np.linspace(4291,4558-1,4558-4291).astype(int)
for i in range (0,1):
    fig, axs=plt.subplots(k,2)
    #fig.suptitle("Patient "+str(patient)+", Seizure "+str(i+3+1) + "\n 1st Peak at 14h00")
    axs[0,0].set_title('Real Labels')
    axs[0,1].set_title('Predicted Labels')
    for j in range(0,k):
        indexes_labels_1=np.where(np.array(pre_ictal_labels_scatter[i])==1)
        indexes_labels_0=np.linspace(0,len(fp_scatter[i])-1,len(fp_scatter[i]))
        indexes_labels_0=np.delete(indexes_labels_0,indexes_labels_1).astype(int)
        
        indexes_predicted_1=np.where(np.array(classification_scatter[i])==1)
        indexes_predicted_0=np.linspace(0,len(fp_scatter[i])-1,len(fp_scatter[i]))
        indexes_predicted_0=np.delete(indexes_predicted_0,indexes_predicted_1).astype(int)
        
        axs[j,0].scatter(feature_data[i][indexes_labels_0,j],np.array(fp_scatter[i])[indexes_labels_0],
           marker='o', alpha=0.5, c="gainsboro")  
        
        axs[j,0].scatter(feature_data[i][indexes_labels_1,j],np.array(fp_scatter[i])[indexes_labels_1],
           marker='x', alpha=0.5, c="steelblue")
        
        #axs[j,0].set_title(feature_names_ICE[j], fontsize=8)
        
        if j!=k-1:
            axs[j,0].get_xaxis().set_visible(False)
            
        axs[j,0].set_xlim(-2,8)
        
        
        
        axs[j,1].scatter(feature_data[i][indexes_predicted_0,j],np.array(fp_scatter[i])[indexes_predicted_0],
           marker='o', alpha=0.5, c="gainsboro")  
        
        axs[j,1].scatter(feature_data[i][indexes_predicted_1,j],np.array(fp_scatter[i])[indexes_predicted_1],
           marker='x',alpha=0.5, c="steelblue")
        
        axs[j,1].scatter(feature_data[i][selected_samples,j],np.array(fp_scatter[i])[selected_samples],
           marker='x',alpha=0.2, color="k")
        
        axs[j,1].set(xlabel='Feature Value \n Predicted', ylabel='FP')
        axs[j,0].set(xlabel='Feature Value \n True', ylabel='FP') 
        
        if j!=k-1:
            axs[j,1].get_xaxis().set_visible(False)
            
        axs[j,1].set_xlim(-2,8)
        
        #axs[j,1].set_title(feature_names_ICE[j], fontsize=8)
          
    
# Counterfactual explanations

# seizure 4
# how to explain the 1st peak, at 14h00, that went until FP=0.31?
    #from samples 4291 to 4558
counter_factual_explanations=[]
for i in range(4291,4558):
    counter_factual_explanations.append(counterFactualExplanation(feature_data[0][i],log_reg))

title=("Patient "+str(patient)+", Seizure "+str(0+3+1) + "\n 1st Peak at 14h00")
[names, counter_value, counter_std]=analyzeCounterFactualExplanations(counter_factual_explanations, feature_names_ICE,title)

print("For the prediction to be different, we would have to change the following features (top 3 by order of importance):")
for i in range(0,3):
    print(names[i]+": "+str(np.round(counter_value[i],2))+"+/-"+str(np.round(counter_value[i],2)))

            
