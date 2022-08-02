# code for the CNNs for all patients
from train_onePatient_DL import train_model_DeepLearning
import numpy as np

patients=[8902]


sops=np.zeros([len(patients)])

for i in range (0,len(patients)):
   print ("Calculating for patient " + str(patients[i]))
   sops[i]=train_model_DeepLearning(patients[i])
   np.save('where do you want to save the preictal periods',sops)
   
   