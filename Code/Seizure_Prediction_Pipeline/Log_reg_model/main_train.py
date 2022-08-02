# Logistic Regression Pipeline: training phase
# to find the optimal pre-ictal period and feature number
# this code will print the optimal preictal period and feature number, for each patient

# mauro pinto

from train_onePatient_logReg import calculatePreIctalAndFeatureNumber 

patients=[8902];

for i in range (0,len(patients)):
   print ("Performing grid search for patient " + str(patients[i]))
   [preictal_period,k_features]=calculatePreIctalAndFeatureNumber(patients[i])
