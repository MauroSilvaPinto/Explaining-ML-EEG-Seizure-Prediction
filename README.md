## Explaining-ML-EEG-Seizure-Prediction
This is the code used for the paper "Explaining Machine Learning Models for EEG Seizure Prediction". It provides three different full Machine Learning pipelines for seizure prediction (class-balanced Logistic Regression Model, ensemble of 15 Support Vector Machines (SVMs), and an ensemble of 3 Convolutional Neural Networks (CNNs)), including surrogate analysis. It also provides a series of explainability methods adapted to EEG seizure prediction.

You can not execute these codes as it is necessary the preprocessed data from EEG recordings. As the used dataset belongs to EPILEPSIAE, we can not make it publicly available online due to ethical concerns. We can only offer the extracted features from non-overlapping windows of 5 seconds, for patient 8902, upon reasonable request. These patient 9802's features are not available on Github due to their memory size.

## Code Folders
- Feature_Extraction
- Seizure_Prediction_Pipelines
- Explanations

## Data Folders
- CNNs
- Interviews_presentations
- Log_Reg_time_plots


## Feature Extraction
- [main_feature_extraction.m] - this is the code you need to execute and adapt. This code contains the master function, to extract all features, for all the signal from a given patient.
- [univariate_linear_features] - a code to extract all features from a given window of data. This code will call the remaining feature extraction functions.
- [spectral_edge.m] - a code to extract several features related to spectral edge power and spectral edge frequency.
- [freqband_power] - a code to calculate the power in a frequency band.
- [discr_wavelet_trans.m] - a code to perform wavelet decomposition.

## Seizure Prediction Pipelines
3 folders are available, one for each pipeline: Logistic Regression, Ensemble of 15 SVMs, and Ensemble of 3 CNNs.
- [main_train] - execute it to train a model and/or to get the grid-search parameters (preictal period, k number of features, SVM C value).
- [main test] - test the model in new seizures and get the performance (seizure sensitivity, FPR/h, and surrogate analysis).

## Explainability Methods



## More Information
59
More information, please just check the article at:
60
​
61
​
62
You are free to use any of this material. If you do, please just cite it as:
63
Pinto, M., Coelho, T., Leal, A. et al. Interpretable EEG seizure prediction using a multiobjective evolutionary algorithm. Sci Rep 12, 4420 (2022). https://doi.org/10.1038/s41598-022-08322-w
64
​
