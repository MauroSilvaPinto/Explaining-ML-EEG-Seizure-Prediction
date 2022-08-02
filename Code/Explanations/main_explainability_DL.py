# code for explainability for one CNN
# 1) Loading the network
# 2) load the data and sleep-awake states
# 3) Get classifications over time
# 4) Classifier time plots
# 5) moment of interest (LIME)
# 6) save in edf to clinicians


# pip install Keras==2.4.3
# tensorflow-gpu                     2.4.1 
import matplotlib.pyplot as plt
import matplotlib.dates as md
from tensorflow import keras

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Run on CPU
import numpy as np

os.chdir("Ensemble_tree_CNNs")
import utils
import pyedflib

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
import tensorflow as tf
from keras.utils import to_categorical

patient=8902
total_seizures=5
sop=50

#%% 1) Loading the network

#the network architecture   
swish_function = tf.keras.activations.swish
nr_filters = 32
filter_size = 5
     
input_layer=Input(shape=(1280,19,1))
        
x=Conv2D(nr_filters,(filter_size,3),(1,1),'same')(input_layer)
x=Conv2D(nr_filters,(filter_size,3),(2,2),'same')(x)
x=SpatialDropout2D(0.5)(x)
x=Activation(swish_function)(x)
x=BatchNormalization()(x)
        
x=Conv2D(nr_filters*2,(filter_size,3),(1,1),'same')(x)
x=Conv2D(nr_filters*2,(filter_size,3),(2,2),'same')(x)
x=SpatialDropout2D(0.5)(x)
x=Activation(swish_function)(x)
x=BatchNormalization()(x)
        
x=Conv2D(nr_filters*4,(filter_size,3),(1,1),'same')(x)
x=Conv2D(nr_filters*4,(filter_size,3),(2,2),'same')(x)
x=SpatialDropout2D(0.5)(x)
x=Activation(swish_function)(x)
x=BatchNormalization()(x)
        
x=GlobalAveragePooling2D()(x)    
x=Dropout(0.5)(x)   
x=Dense(2)(x)
        
output_layer=Activation('softmax')(x)
model=Model(input_layer,output_layer)

os.chdir("Results_mauro")
os.chdir("Patient "+str(patient))

model.load_weights(f'seizure_prediction_model_0.h5')
norm_values = np.load(f'norm_values_0.npy')

#%% 2) load the data and sleep-awake states

# where the data is
path=r"D:\Doentes_mauro"
# go to where the data is
os.chdir(path)
os.chdir("pat_"+str(patient)+"_splitted")

# load seizure onset's
seizure_information=np.load("all_seizure_information.pkl",allow_pickle=True)

## loading vigilance vectors
vigilance=np.load("pat_"+str(patient)+"_vigilance",allow_pickle=True)
vigilance_datetimes=np.load("pat_"+str(patient)+"_datetimes",allow_pickle=True)

sph=10
window_length=5

for i in range(0,len(vigilance)):
    vigilance[i]=np.abs(vigilance[i]-1)
    vigilance[i]=np.clip(vigilance[i],0.05,0.95)
     
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

#%% 3) Get classifications over time

#################### Classification ###########################
predicted_labels=[]
        
for i in range(0,len(testing_labels)):
    current_testing_data = (testing_data[i]-norm_values[0])/norm_values[1]
    y_pred = model.predict(current_testing_data)
    y_pred = np.argmax(y_pred,axis=1)
    predicted_labels.append(y_pred)
    
    
###################### Classification Smooth #######################
firing_power_smooth=[]
for i in range(0,len(testing_labels)): 
    firing_power_smooth.append(utils.SmoothOutput(predicted_labels[i],sop,window_length))

###################### Firing Power + Refractory #############
for i in range(0,len(testing_labels)):
    predicted_labels[i]=utils.FiringPowerAndRefractoryPeriod(predicted_labels[i],testing_datetimes[i],sop,sph,window_length)  
    
############# changing the variable names ######################################

datetimes=testing_datetimes
labels=testing_labels
alarms=predicted_labels
firing_power_output=firing_power_smooth
data=testing_data
fp_scatter=firing_power_output.copy()

del testing_data
del testing_labels
del testing_onsets
del current_testing_data
del filter_size
del firing_power_smooth
del nr_filters
del seizure_labels
del seizure_onset
del seizure_k

#%% 4) Classifier time plots

import datetime

# re-arrange datetimes
new_datetimes=[]
for i in range(0,total_seizures-3):
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
    
    for j in range(0,len(datetimes[i])-1):
        time_difference=datetimes[i][j+1]-datetimes[i][j]
        time_difference=time_difference.seconds
    
        datetimes_new_i.append(datetimes[i][j])
        firing_power_output_new_i.append(firing_power_output[i][j])
        labels_new_i.append(labels[i][j])
        alarms_new_i.append(alarms[i][j])
            
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
    
    datetimes_new.append(datetimes_new_i)
    firing_power_output_new.append(firing_power_output_new_i)
    labels_new.append(labels_new_i)
    alarms_new.append(alarms_new_i)
    

datetimes=datetimes_new
firing_power_output=firing_power_output_new
labels=labels_new
alarms=alarms_new


# plotting Firing Power output throughout time

import datetime

for i in range (0,total_seizures-3):
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
    
    
    plt.title("Patient "+str(patient)+", Seizure "+str(i+3+1))

#%% 5) moment explanations
    # in this case, for example, the Peak from seizure 4 at 14h

#       
## plotting Firing Power output throughout time
## check the samples i need to retrieve manually
#for i in range (0,total_seizures-3):
#     plt.figure()  
#
#     plt.plot(fp_scatter[i],'k',alpha=0.7)
#     plt.plot(np.linspace(0.7, 0.7, len(fp_scatter[i])),linestyle='--',
#             color='black',alpha=0.7)
#  
#     plt.title("Patient "+str(patient)+", Seizure "+str(i+3+1))
    
# Seizure 4
# Point A:
    # from 00h30 to  00h59
    # samples: 11621 - 11919

    
beginning_sample=11621
ending_sample=11919
seizure=0 # 0 because it is the first testing seizure
seizure_to_analyze=data[seizure]

moment_description="point_A_DL"
  
# LIME #######
from lime import lime_image
from skimage.segmentation import mark_boundaries


def predict(images):
    images = images[:,:,:,1]
    images = images[...,np.newaxis]
    return model.predict(images)

     
seizure_to_analyze=data[seizure]
datetime_seizure=datetimes[seizure]

image_raw=[]
mask=[]

for s in range(beginning_sample,ending_sample):
    sample_to_analyze=s
    c=2
    explainer = lime_image.LimeImageExplainer()
    img_array = seizure_to_analyze[s,:,:]
    
    img_array_raw_i=img_array
    img_array=np.repeat(img_array_raw_i,3,axis=2)
    img_array = keras.preprocessing.image.img_to_array(img_array)
    
    explanation = explainer.explain_instance(img_array, predict,  
                                             top_labels=3, hide_color=0, num_samples=250)
    
    
    
    temp, mask_i = explanation.get_image_and_mask(explanation.top_labels[0],
                                                    positive_only=True, num_features=5, hide_rest=True)

    
    image_raw.append(img_array_raw_i)
    mask.append(mask_i)
    
    
image_raw=np.squeeze(np.concatenate(image_raw),axis=2)
mask=np.concatenate(mask)

from matplotlib.widgets import Slider
from matplotlib.widgets import RadioButtons

global window_size_seconds
window_size_seconds=30
fs=256

electrodes=["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2",
            "F7","F8","T7","T8","P7","P8","FZ","CZ","PZ"]
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

# Plotting LIME results
for i in range(0,19):
    eeg_channel=image_raw[:,i]
    eeg_channel=(eeg_channel-np. min(eeg_channel)) / (np. max(eeg_channel) - np. min(eeg_channel))
    ax.plot(np.linspace(0,np.shape(image_raw)[0]-1,np.shape(image_raw)[0]),eeg_channel+i-0.5,'k')
    
    eeg_mask=mask[:,i]
    eeg_mask = eeg_mask.astype('float')
    eeg_mask[eeg_mask==0]=np.nan
    
    mask_x=[]
    mask_y=[]
    for j in range(0,len(eeg_mask)):
        if np.isnan(eeg_mask[j])==False:
            mask_x.append(j)
            mask_y.append(eeg_channel[j]+i-0.5)
    ax.plot(mask_x,mask_y,'r.',alpha=0.10,markersize=10)

ax.set_xlim([0, np.shape(image_raw)[0]])
ax.set_yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
ax.set_yticklabels(electrodes)
ax.set_xticks(np.arange(0,np.shape(image_raw)[0],256))
ax.set_xticklabels(np.arange(0,int(np.shape(image_raw)[0]/256)))
plt.ylabel("electrodes")
plt.xlabel("time (s)")

axcolor = 'lightgoldenrodyellow'
axpos = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)

spos = Slider(axpos, 'Pos', 0.1, np.shape(image_raw)[0])

def update(val):
    global window_size_seconds

    pos = spos.val
    ax.axis([pos,pos+window_size_seconds*fs,-1,19])
    fig.canvas.draw_idle()
    
axcolor = 'lightgoldenrodyellow'
rax = plt.axes([0.05, 0.7, 0.05, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('5s', '15s', '30s', '60s'),active=2)


def hzfunc(label):
    global window_size_seconds
    hzdict = {'5s': 5, '15s': 15,'30s': 30, '60s':60}
    pos = spos.val
    window_size_seconds = hzdict.__getitem__(label)
    ax.axis([pos,pos+int(window_size_seconds)*fs,-1,19])
    fig.canvas.draw_idle()
        
radio.on_clicked(hzfunc)
spos.on_changed(update)
plt.show()

#%% 6) save in edf to clinicians

## Annotations
mask=np.sum(mask,axis=1)
mask=np.clip(mask, 0, 1)    

fs=256

electrodes=["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2",
            "F7","F8","T7","T8","P7","P8","FZ","CZ","PZ"]

f = pyedflib.EdfWriter(str(patient)+"_seizure_"+str(seizure+4)+"_"+moment_description+".edf", 19,
                           file_type=pyedflib.FILETYPE_EDFPLUS)

channel_info = []
data_list = []

for i in range(0,19):
    ch_dict = {'label':electrodes[i], 'dimension': 'uV', 'sample_frequency': fs,
               'physical_max': 100, 'physical_min': -100, 'digital_max': 32767,
               'digital_min': -32768, 'transducer': '', 'prefilter':''}
    data_list.append(image_raw[:,i])
    channel_info.append(ch_dict)
    

indexes_beginning=np.argwhere(np.diff(mask)==1)
indexes_ending=np.argwhere(np.diff(mask)==-1)

f.setSignalHeaders(channel_info)
f.writeSamples(data_list)

# i can't save all annotations
# so let's save annotations that have size that make sense for a clinician

# an eeg spike lasts from 20-70ms.
# so i will drop annotations with less size than 20ms

annotations=np.zeros((len(indexes_ending),2))
for i in range(0,len(indexes_beginning)):
    annotations[i,0]=indexes_beginning[i][0]/256
    annotations[i,1]=(indexes_ending[i]-indexes_beginning[i])[0]/256
    
indexes_sorted=np.flip(np.argsort(annotations[:,1],axis=0))
annotations=annotations[indexes_sorted,:]
    

for i in range(0,len(indexes_beginning)):
    duration=annotations[i,1]
    beginning=annotations[i,0]
    # durations higher than 20ms
    if (duration>0.02):
        f.writeAnnotation(beginning,duration,"e")
    
f.close()

del f

file = pyedflib.EdfReader(str(patient)+"_seizure_"+str(seizure+4)+"_"+moment_description+".edf")  
annotations = file.readAnnotations()  
print(annotations)  
file.close()
