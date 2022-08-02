# a code to test all patients

import numpy as np
from test_onePatient_logReg import testOnePatient 


performance=np.zeros([1,6])
patients=[8902];
seizures=[5];
sop=[25]
k=[7]


for i in range (0,len(patients)):
   print ("Calculating for patient " + str(patients[i]))
   performance[i,:]=testOnePatient(patients[i],sop[i],k[i],seizures[i])
   
