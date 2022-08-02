
function feature_extraction_code(patient,number_of_seizures)

% patients directory
% you should have a patient directory with a folder for each patient, with the following structure:
% pat_[patient_number]_splitted

patients_dir="Insert your patients directory";
cd(patients_dir)


% Inside the folder for each patient, you should have several files, two for each seizure with the following structure:
% i) seizure_[seizure_number]_data.npy
	% each file should have the following structure:
	% N windows of 5 seconds*1280 samples * 19 channels
	% 1280 dimensions is due to 5 seconds windows of 256Hz of frequency sampling
	% 19 channels follow the 19-20 system
% ii) feature_datetimes_[seizure_number].npy
    % N windows of 5 seconds



cd(strcat("pat_",num2str(patient),"_splitted"));

% stuff related to the feature extraction
fs=256;
plotFigure=0; % no figures from the feature extraction
psd_method="welch";

% number of total features: 59 linear + 31 nonlinear (all univariate)
total_n_features=59;%+31;

% iterating all seizures
for seizure=0:number_of_seizures-1
    % entering again in the folder files
    cd(patients_dir)
    cd(strcat("pat_",num2str(patient),"_splitted"));
    
    % loading a seizure
    seizure_data = double(readNPY(strcat("seizure_",num2str(seizure),"_data.npy")));
    % building the feature matrix, which is empty
    seizure_data_dimensions=size(seizure_data);
    seizure_features=zeros(seizure_data_dimensions(1),...
        total_n_features,...
        seizure_data_dimensions(3));
    
    % iterating all windows, to extract our dearest features
    for row=1:size(seizure_data,1)
        parfor channel=1:size(seizure_data,3)
            % the current window we are iterating
            window=seizure_data(row,:,channel);
            % extracting linear features
            [linear_features, ~, ~, ~] = ...
                univariate_linear_features(window, fs, psd_method, plotFigure);
            % extracting nonlinear features
            %             [nonlinear_features, ~, ~, ~] = ...
            %                 univariate_nonlinear_features(window, fs, plotFigure);
            % concatenating all features into a vector
            feature_vector=[linear_features];%;nonlinear_features]';
            % storing the feature vector
            seizure_features(row,:,channel)=feature_vector;
        end
    end
    
    % go back
    cd("..")
    % enter in the features folder
    if ~exist(strcat("pat_",num2str(patient),"_features"), 'dir')
        mkdir(strcat("pat_",num2str(patient),"_features"))
        cd(strcat("pat_",num2str(patient),"_features"))
    else
        cd(strcat("pat_",num2str(patient),"_features"))
    end
    % saving seizure file
    writeNPY(seizure_features, strcat("pat_",num2str(patient),"_seizure_",num2str(seizure),"_features.npy"));
end

% signalling it is done
cd(patients_dir)
done=1;
writeNPY(done, strcat("pat_",num2str(patient),"_complete"))

end

%% feature names

% linear_feature_names=["Delta_power","Theta_power","Alpha_power","Beta_power",...
%     "Gamma1_power","Gamma2_power","Gamma3_power","Gamma4_power",...
%     "Relative_delta_power","Relative_theta_power","Relative_alpha_power",...
%     "Relative_beta_power","Relative_gamma1_power","Relative_gamma2_power",...
%     "Relative_gamma3_power","Relative_gamma4_power","Total_power",...
%     "Alpha_peak_frequency","Mean_frequency","Ratio_delta_theta",...
%     "Ratio_delta_alpha","Ratio_delta_beta","Ratio_delta_gamma1",...
%     "Ratio_delta_gamma2","Ratio_delta_gamma3","Ratio_theta_alpha",...
%     "Ratio_theta_beta","Ratio_theta_gamma1","Ratio_theta_gamma2",...
%     "Ratio_theta_gamma3","Ratio_alpha_beta","Ratio_alpha_gamma1",...
%     "Ratio_alpha_gamma2","Ratio_alpha_gamma3","Ratio_beta_gamma1",...
%     "Ratio_beta_gamma2","Ratio_beta_gamma3","Ratio_gamma1_gamma2",...
%     "Ratio_gamma1_gamma3","Ratio_gamma2_gamma3",...
%     "Ratio_beta_over_alpha_theta","Ratio_theta_over_alpha_beta",...
%     "Normalized_mean_intensity","Mean_intensity","Std","Kurtosis",...
%     "Skewness","Activity","Mobility","Complexity","Spectral_edge_frequency",...
%     "Spectral_edge_power","Decorrelation_time","energy_D1","energy_D2",...
%     "energy_D3","energy_D4","energy_D5","energy_A5"];


