# a code to test all patients using the CNNs ensemble

# mauro pinto

import os
import numpy as np
from test_onePatient_DL import testOnePatient 

patients=[8902]
seizures=[5]
performance=np.zeros([len(patients),6])

path_sops='/media/fabioacl/EPILEPSIAE Preprocessed Data/Cose di Mauri/Results_mauro/sops.npy'
sops=np.load(path_sops,allow_pickle=True)

for i in range (0,len(patients)):
   print ("Calculating for patient " + str(patients[i]))
   performance[i,:]=testOnePatient(patients[i],sops[i],seizures[i])
 
# where I save the performance
path=root_path = "/media/fabioacl/EPILEPSIAE Preprocessed Data/Cose di Mauri/Results_mauro/"
# go to where the data is
os.chdir(path)   
np.save("test_performance_DL.npy", performance)
