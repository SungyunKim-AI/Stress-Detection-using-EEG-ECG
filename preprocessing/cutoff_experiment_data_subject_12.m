% ========================== Cut-off Data =================================
% baseline : 눈 뜨고 15초, 눈 감고 15초 총 40초 잘라내기 → 대략 1분 20초
% stimuli : 앞 부분 5초, 뒷부분 2초 제거 → 대략 1분 8초 남는다.
% experimentTime : 실험 시간 (초 단위)
% restEnd : 휴식 종료 시간 (초 단위)
% cptStart : CPT 시작 시간 (초 단위)

subject = 12;
experimentTime = [195, 195, 195, 195, 195, 195, 195, 195, 195, 195];
restEnd =        [120, 120, 120, 120, 120, 120, 120, 120, 120, 120];
cptStart =       [122, 122, 122, 122, 122, 122, 122, 122, 122, 122];

noOfSamples = 10;
SamplingRate_ECG = 512;      % Emotive EpocX Sampling Rate (Hz 단위)

load_path_EEG = "C:\\Users\\user\\Desktop\\Experiment Data\\EEG\\";
load_path_ECG = "C:\\Users\\user\\Desktop\\Experiment Data\\ECG\\";
save_path_EEG = "C:\\Users\\user\\Desktop\\data_preprocessed\\cutoff_preprocessed\\EEG\\";
save_path_ECG = "C:\\Users\\user\\Desktop\\data_preprocessed\\cutoff_preprocessed\\ECG\\";
% =========================================================================

for sample = 5:noOfSamples
    fprintf('========= Sample %d of Subject %d =========\n',sample,subject);
    fprintf('Total Experiment Time %d (s)\n', experimentTime(sample));
    fprintf('CPT Start Time %d (s)\n', cptStart(sample));
    fprintf('CPT Time %d (s)\n', experimentTime(sample) - cptStart(sample));
    
    % ========================= EEG ==================================
    fprintf('=> EEG\n');
    fprintf('********** No Such EEG File :');
    disp("s" + subject + "_" + sample + ".csv");

    
    % ========================= ECG ==================================
    fprintf('=> ECG\n');
    % Data load
    file_path = char(load_path_ECG + "s" + subject + "_" + sample + ".csv");
    try
        ECG_data = readtable(file_path,"VariableNamingRule","preserve");
    catch
        fprintf('********** No Such ECG File :');
        disp("s" + subject + "_" + sample + ".csv");
        continue;
    end
               
    % Data Cut-off
    experimentTime(sample) = experimentTime(sample) * SamplingRate_ECG;
    exData = ECG_data{1:experimentTime(sample), 4:6};  % table 데이터를 matrix 데이터로 변경  
    restEnd(sample) = restEnd(sample) * SamplingRate_ECG;
    cptStart(sample) = cptStart(sample) * SamplingRate_ECG;

    % baseline : 앞 부분 30초, 뒷 부분 5초 제거
    baseline_ecg = exData(1:restEnd(sample),:);
    baseline_ecg(1:(30*SamplingRate_ECG), :) = []; 
    baseline_ecg(end:-(5*SamplingRate_ECG), :) = [];
    
    % stimuli : 앞 부분 5초, 뒷 부분 2초 제거
    stimuli_ecg = exData(cptStart(sample):end, :);
    stimuli_ecg((5*SamplingRate_ECG):-(2*SamplingRate_ECG), :) = [];
    timeStamp_eeg = [baseline_ecg(1,1), baseline_ecg(end,1), stimuli_ecg(1,1), stimuli_ecg(end,1)]; %(s)단위

    fprintf('ECG baseline size :');
    disp(size(baseline_ecg));
    fprintf('ECG stimuli size :');
    disp(size(stimuli_ecg));

    % Save csv file
    filename = char(save_path_ECG + "baseline\\s" + subject + "_" + sample + ".csv");
    writematrix(baseline_ecg, filename);
    filename = char(save_path_ECG + "stimuli\\s" + subject + "_" + sample + ".csv");
    writematrix(stimuli_ecg, filename);
end