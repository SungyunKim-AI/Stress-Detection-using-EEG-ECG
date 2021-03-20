% ===================== Preprocessing Experiment Data =====================
% [공통 과정]
% Hamming sinc linear phase FIR filters 적용
% Remove DC offset
% Shift by the filter’s group delay
% [선택 과정]
% 1. CAR (Common Average Reference)
% 2. ASR (Artifact Subspace Reconstruction) + CAR
% 3. CAR + FFT
% 4. ASR + CAR + FFT

noOfSubjects = 5;           % 실험 대상 수
noOfSamples = 10;           % 실험 수
samplingRate_EEG = 128;     % EEG Sampling Rate (Hz)
samplingRate_ECG = 51.2;    % ECG Sampling Rate (Hz)

% Bandpass Filter
% 1: overall, 2: theta, 3: alpha, 4: beta 
FilterOrder = 212;      
Filter = {[30 4], [8 4], [13 8], [30 12]};     
FilterName = {'overall', 'theta', 'alpha', 'beta'};

load_path_EEG = "C:\\Users\\user\\Desktop\\data_preprocessed\\cutoff_preprocessed\\EEG\\";
load_path_ECG = "C:\\Users\\user\\Desktop\\data_preprocessed\\cutoff_preprocessed\\EEG\\";
save_path_EEG = "C:\\Users\\user\\Desktop\\data_preprocessed\\band_filter_preprocessed\\EEG";
save_path_ECG = "C:\\Users\\user\\Desktop\\data_preprocessed\\band_filter_preprocessed\\ECG";
eeglab;
%==========================================================================

for sub = 1:noOfSubjects
    for sample = 1:noOfSamples
        % => baseline
        % Data import
        fileName = char(load_path_EEG + "baseline\\s" + sub + "_" + sample + ".csv");
        baseline = readtable(fileName);
        baseline = baseline{:,:}.';
        baseline_EEG = pop_importdata('dataformat','array','nbchan',0,'data','baseline','srate',samplingRate_EEG,'pnts',0,'xmin',0);
        
        % Hamming sinc linear phase FIR filters
        baseline_EEG = pop_eegfiltnew(baseline_EEG, 'locutoff',4,'hicutoff',30,'plotfreqz',0);
        
        % 1. CAR (Common Average Reference)
        baseline_EEG = pop_reref(baseline_EEG, []);
        
        % => stimuli
        fileName = char(load_path_EEG + "stimuli\\s" + sub + "_" + sample + ".csv");
        stimuli_EEG = readtable(fileName);
        
    end
end

