# a code that shows the constructed explanations
# this code is organised as follows:
    # 1) getting the Firing Power, datetimes, and vigilance state vectors
    # 2) get the classifier time plots
# as the others steps are equal to the ones in log reg, we do not put these here
# if you want to see them, please see main_explainability_logReg.py

# mauro pinto
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime
import numpy as np
import utils

from test_OnePatient_getPlots_SVMs import testOnePatientGetPlots 

#%% 1) getting the Firing Power, datetimes, and vigilance state vectors

patient=53402
seizures=4
sop=55
k=3
c=2**(-10)

            
[datetimes, seizure_datetime, labels, alarms,
 firing_power_output, firing_power_all, feature_names]=testOnePatientGetPlots(patient,sop,k,seizures,c)


vigilance=np.load("pat_"+str(patient)+"_vigilance",allow_pickle=True)
vigilance_datetimes=np.load("pat_"+str(patient)+"_datetimes",allow_pickle=True)

for i in range(0,len(vigilance)):
    vigilance[i]=np.abs(vigilance[i]-1)
    vigilance[i]=np.clip(vigilance[i],0.05,0.95)






#%% 2) get the classifier time plots
    
# re-arrange data for plotting
# re-arrange datetimes
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

firing_power_all_new=[]
# filling the missing data
for i in range(0,len(datetimes)):
    labels[i]=labels[i].tolist()
    firing_power_output[i]=firing_power_output[i].tolist()
    
    datetimes_new_i=[]
    firing_power_output_new_i=[]
    labels_new_i=[]
    alarms_new_i=[]
    
    firing_power_all_new_i=[[] for i in range(15)]    
    for j in range(0,len(datetimes[i])-1):
        time_difference=datetimes[i][j+1]-datetimes[i][j]
        time_difference=time_difference.seconds
    
        datetimes_new_i.append(datetimes[i][j])
        firing_power_output_new_i.append(firing_power_output[i][j])
        labels_new_i.append(labels[i][j])
        alarms_new_i.append(alarms[i][j])
        
        # iterate the 15 classifiers
        for k in range(0,15):
            firing_power_all_new_i[k].append(firing_power_all[i][k][j])    
            
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
                
                # iterate the 15 classifiers
                for k in range(0,15):
                    firing_power_all_new_i[k].append(np.NaN)

    
    datetimes_new.append(datetimes_new_i)
    firing_power_output_new.append(firing_power_output_new_i)
    labels_new.append(labels_new_i)
    alarms_new.append(alarms_new_i)
    firing_power_all_new.append(firing_power_all_new_i)
    

datetimes=datetimes_new
firing_power_output=firing_power_output_new
labels=labels_new
alarms=alarms_new

for i in range (0,seizures-3):
    plt.figure()  

    plt.plot(datetimes[i],firing_power_output[i],'k',alpha=0.7)
    plt.plot(datetimes[i],np.linspace(0.7, 0.7, len(datetimes[i])),linestyle='--',
             color='black',alpha=0.7)
   
    plt.grid()
    
    plt.ylim(0,1)
    plt.xlim(datetimes[i][0],datetimes[i][len(datetimes[i])-1])
    
    for k in range(0,15):
        plt.plot(datetimes[i],firing_power_all_new[i][k], color='black',alpha=0.15)
    
    
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
    
    
    plt.title("Patient "+str(patient)+", Seizure "+str(i+3+1))
            
