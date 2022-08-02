# ensemble of SVMs. code to train all patients
# this code finds, for each patient:
	#the best preictal period, number of features, and C-value

import numpy as np
from train_onePatient_SVMs import calculatePreIctalAndFeatureNumber 

patients=[8902];

for i in range (0,len(patients)):
   print ("Calculating for patient " + str(patients[i]))
   [preictal_period,k_features]=calculatePreIctalAndFeatureNumber(patients[i])

   