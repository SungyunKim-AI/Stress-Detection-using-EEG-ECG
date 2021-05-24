subject = 5;
sample = 4;

% ========================== EEG ==========================
% Raw data of EEG
path = "C:\\Users\\user\\Desktop\\data_preprocessed\\cutoff_preprocessed\\EEG\\";
path_EEG_baseline = char(path + "baseline\\s" + subject + "_" + sample + ".csv");
data_EEG_baseline = readtable(path_EEG_baseline,"VariableNamingRule","preserve");
path_EEG_stimuli = char(path + "stimuli\\s" + subject + "_" + sample + ".csv");
data_EEG_stimuli = readtable(path_EEG_stimuli,"VariableNamingRule","preserve");

% Bandpass filter data of EEG
path = "C:\\Users\\user\\Desktop\\data_preprocessed\\band_filter_preprocessed\\EEG\\";
path_filted_EEG_baseline = char(path + "baseline\\alpha\\s" + subject + "_" + sample + ".csv");
data_filted_EEG_baseline = readtable(path_filted_EEG_baseline,"VariableNamingRule","preserve");
path_filted_EEG_stimuli = char(path + "stimuli\\alpha\\s" + subject + "_" + sample + ".csv");
data_filted_EEG_stimuli = readtable(path_filted_EEG_stimuli,"VariableNamingRule","preserve");

% ASR + CAR filter data of EEG
path = "C:\\Users\\user\\Desktop\\data_preprocessed\\ASR_CAR_preprocessed\\EEG\\";
path_ASR_EEG_baseline = char(path + "baseline\\alpha\\s" + subject + "_" + sample + ".csv");
data_ASR_EEG_baseline = readtable(path_ASR_EEG_baseline,"VariableNamingRule","preserve");
path_ASR_EEG_stimuli = char(path + "stimuli\\alpha\\s" + subject + "_" + sample + ".csv");
data_ASR_EEG_stimuli = readtable(path_ASR_EEG_stimuli,"VariableNamingRule","preserve");

% EEG Data Figure
f1 = figure('Name','EEG baseline preprocessing','NumberTitle','off');
subplot (3,1,1), plot(data_EEG_baseline{:,:}), title ('EEG baseline Signal with Noise'), grid on
subplot (3,1,2), plot(data_filted_EEG_baseline{:,:}.'), title ('EEG baseline Signal with Bandpass filter'), grid on
subplot (3,1,3), plot(data_ASR_EEG_baseline{:,:}.'), title ('EEG baseline Signal with ASR, CAR filter'), grid on

f2 = figure('Name','EEG stimuli preprocessing','NumberTitle','off');
subplot (3,1,1), plot(data_EEG_stimuli{:,:}), title ('EEG stimuli Signal with Noise'), grid on
subplot (3,1,2), plot(data_filted_EEG_stimuli{:,:}.'), title ('EEG stimuli Signal with Bandpass filter'), grid on
subplot (3,1,3), plot(data_ASR_EEG_stimuli{:,:}.'), title ('EEG stimuli Signal with ASR, CAR filter'), grid on


% ========================== EEG ==========================
% Raw data of ECG
path = "C:\\Users\\user\\Desktop\\data_preprocessed\\cutoff_preprocessed\\ECG\\";
path_ECG_baseline = char(path + "baseline\\s" + subject + "_" + sample + ".csv");
data_ECG_baseline = readtable(path_ECG_baseline,"VariableNamingRule","preserve");
path_ECG_stimuli = char(path + "stimuli\\s" + subject + "_" + sample + ".csv");
data_ECG_stimuli = readtable(path_ECG_stimuli,"VariableNamingRule","preserve");

% Bandpass filter data of EEG && Down sampling to 128Hz
path = "C:\\Users\\user\\Desktop\\data_preprocessed\\band_filter_preprocessed\\ECG_128\\";
path_filted_ECG_baseline = char(path + "baseline\\s" + subject + "_" + sample + ".csv");
data_filted_ECG_baseline = readtable(path_filted_ECG_baseline,"VariableNamingRule","preserve");
path_filted_ECG_stimuli = char(path + "stimuli\\s" + subject + "_" + sample + ".csv");
data_filted_ECG_stimuli = readtable(path_filted_ECG_stimuli,"VariableNamingRule","preserve");

% ECG Data Figure
f3 = figure('Name','ECG baseline preprocessing','NumberTitle','off');
subplot (2,1,1), plot(data_ECG_baseline{:,:}), title ('ECG baseline Signal with baseline wander'), grid on
subplot (2,1,2), plot(data_filted_ECG_baseline{:,:}), title ('EEG baseline Signal with Bandpass filter && Down sampling to 128Hz'), grid on

f4 = figure('Name','ECG stimuli preprocessing','NumberTitle','off');
subplot (2,1,1), plot(data_ECG_stimuli{:,:}), title ('ECG stimuli Signal with baseline wander'), grid on
subplot (2,1,2), plot(data_filted_ECG_stimuli{:,:}), title ('EEG stimuli Signal with Bandpass filter && Down sampling to 128Hz'), grid on
