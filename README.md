## Explaining-ML-EEG-Seizure-Prediction
This is the code used for the paper "Explaining Machine Learning Models for EEG Seizure Prediction". It provides three different full Machine Learning pipelines for seizure prediction (class-balanced Logistic Regression Model, ensemble of 15 Support Vector Machines (SVMs), and an ensemble of 3 Convolutional Neural Networks (CNNs)), including surrogate analysis. It also provides a series of explainability methods adapted to EEG seizure prediction.

You can not execute these codes as it is necessary the preprocessed data from EEG recordings. As the used dataset belongs to EPILEPSIAE, we can not make it publicly available online due to ethical concerns. We can only offer the extracted features from non-overlapping windows of 5 seconds, for patient 8902, upon reasonable request. These patient 9802's features are not available on Github due to their memory size.

## Code Folders
- Feature_Extraction
- Seizure_Prediction_Pipelines
- Explanations

## Data Folders
- CNNs: the trained CNN for patient 8902.
- Interviews_presentations: the presentation slides to make the interviews for clinicians and data scientists.
- Log_Reg_time_plots: all Log. Reg. classifier plots for all patients.

## Feature Extraction
- [main_feature_extraction.m]: this is the code you need to execute and adapt. This code contains the master function, to extract all features, for all the signal from a given patient.
- [univariate_linear_features.m]: a code to extract all features from a given window of data. This code will call the remaining feature extraction functions.
- [spectral_edge.m]: a code to extract several features related to spectral edge power and spectral edge frequency.
- [freqband_power]: a code to calculate the power in a frequency band.
- [discr_wavelet_trans.m]: a code to perform wavelet decomposition.

## Seizure Prediction Pipelines
3 folders are available, one for each pipeline: Logistic Regression, Ensemble of 15 SVMs, and Ensemble of 3 CNNs.
#### The structure is similar in all:
- [main_train.py] - execute it to train a model and/or to get the best grid-search parameters (preictal period, k number of features, SVM C value).
- [main test.py] - test the model in new seizures and get the performance (seizure sensitivity, FPR/h, and surrogate analysis)

#### Specific functions for each pipeline:
- [train_onePatient_logReg.py]: get the best grid-search parameters from the ensemble of Log. Reg. (preictal period, k number of features).
- [train_onePatient_SVMs.py]: get the best grid-search parameters from the ensemble of 15 SVMs (preictal period, k number of features, SVM C value).
- [train_onePatient_CNNs.py]: get the best grid-search parameters from the ensemble of 3 CNNs (preictal period) and saves the correspondent networks.
- [train_CNN.py]: construct the CNN architecture, train it and save it.
- [test_onePatient_logReg.py]: test the Logistic Regression in new seizures.
- [test_onePatient_SVMs.py]: test the Ensemble of 15 SVMs in new seizures.
- [test_onePatient_CNNs.py]: test the Ensemble of 3 CNNs in new seizures.

#### General files for all pipelines
- [utils.py]: code with utility functions.
- [channel_names.pkl]: file with the channels' names by their used order.

## Explainability Methods

- [main_explainability_logReg.py]: get all the developed explanations: classifier time plots, feature influence with Log. Reg. coefficients, Partial Dependence Plots (PDPs), Beeswarm summary of SHAP values, Calibration/scatter curves/plots of all features, Counterfactual explanations of moments of interest.
- [main_explainability_SVMs.py]: get the classifier time plots for an Ensemble of models (SVMs, in this case). 
- [main_explainability_DL.py]: get the classifier time plots for a CNN, the LIME explanations (return points of interest to the model), and how to save EEG parts into edf files.
- [counterFactualExplanation.py]: code that developed counterfactual explanations for segments of interest of the EEG.
- [test_OnePatient_getPlots_logReg.py]:  code to get all variables from the Log. Reg. pipeline to make the time plots.
- [test_OnePatient_getPlots_SVMs.py]: code to get all variables from the SVMs pipeline to make the time plots.


## More Information
More information, please just check the article at:
Still not published, peeps!


You are free to use any of this material. If you do, please just cite it as:
Let's wait :)
