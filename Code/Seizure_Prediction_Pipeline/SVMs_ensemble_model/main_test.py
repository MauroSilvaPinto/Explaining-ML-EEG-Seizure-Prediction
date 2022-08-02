# a code to test the SVMs for all patients

import numpy as np
from test_onePatient_SVMs import testOnePatient 


performance=np.zeros([1,6])
patients=[8902];
seizures=[5];
sop=[20]
k=[30]
c_value=[2**(-10)]


for i in range (0,len(patients)):
   print ("Calculating for patient " + str(patients[i]))
   performance[i,:]=testOnePatient(patients[i],sop[i],k[i],seizures[i],c_value[i])
   