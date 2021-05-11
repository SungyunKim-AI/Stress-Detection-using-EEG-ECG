% ========================== Cut-off Data  =================================
% baseline : 앞 부분 30초, 뒷 부분 5초 제거 → 약 85초
% stimuli : 앞 부분 5초, 뒷 부분 2초 제거   → 약 68초
% experimentTime : 전체 실험 시간 (초 단위)
% cptStart : CPT 시작 시간 (초 단위)

% => 1st Experiment 
% subeject = 1~4 (4명)
% SamplingRate_EEG = 128 Hz
% SamplingRate_ECG = 51.2 Hz

% => 2nd Experiment
% subeject = 5~9 (5명)
% SamplingRate_EEG = 128 Hz
% SamplingRate_ECG = 512 Hz

% => 3rd Experiment
% subeject = 10~19 (10명)
% SamplingRate_EEG = 128 Hz
% SamplingRate_ECG = 512 Hz

% 피실험자 실험 시간 저장 파일 : experiment_time.txt
subject = 19;
experimentTime = [195, 195, 195, 195, 195, 195, 195, 195, 195, 195];
restEnd =        [120, 120, 120, 120, 120, 120, 120, 120, 120, 120];
cptStart =       [122, 122, 122, 122, 122, 122, 122, 122, 122, 122];

noOfSamples = 10;
SamplingRate_EEG = 128;

load_path_EEG = "C:\\Users\\user\\Desktop\\Experiment Data\\EEG\\";
load_path_ECG = "C:\\Users\\user\\Desktop\\Experiment Data\\ECG\\";
save_path_EEG = "C:\\Users\\user\\Desktop\\data_preprocessed\\cutoff_preprocessed\\EEG\\";
save_path_ECG = "C:\\Users\\user\\Desktop\\data_preprocessed\\cutoff_preprocessed\\ECG\\";
% =========================================================================


for sample = 1:noOfSamples
    fprintf('========= Sample %d of Subject %d =========\n',sample,subject);
    fprintf('Total Experiment Time %d (s)\n', experimentTime(sample));
    fprintf('CPT Start Time %d (s)\n', cptStart(sample));
    fprintf('CPT Time %d (s)\n', experimentTime(sample) - cptStart(sample));
    
    % ========================= EEG ==================================
    fprintf('=> EEG\n');
    % Data load
    file_path = char(load_path_EEG + "s" + subject + "_" + sample + ".csv");
    try
        EEG_data = readtable(file_path,"VariableNamingRule","preserve");
    catch
        fprintf('********** No Such EEG File :');
        disp("s" + subject + "_" + sample + ".csv");
        continue;
    end

    % Data Cut-off
    experimentTime(sample) = experimentTime(sample) * SamplingRate_EEG;
    exData = EEG_data{1:experimentTime(sample), [1,4:17]};  % table 데이터를 matrix 데이터로 변경  
    restEnd(sample) = restEnd(sample) * SamplingRate_EEG;
    cptStart(sample) = cptStart(sample) * SamplingRate_EEG;

    % baseline : 앞 부분 30초, 뒷 부분 5초 제거
    baseline_eeg = exData(1:restEnd(sample),:);
    baseline_eeg(1:(30*SamplingRate_EEG), :) = []; 
    baseline_eeg(end:-(5*SamplingRate_EEG), :) = [];
    
    % stimuli : 앞 부분 5초, 뒷 부분 2초 제거
    stimuli_eeg = exData(cptStart(sample):end, :);
    stimuli_eeg((5*SamplingRate_EEG):-(2*SamplingRate_EEG), :) = [];
    timeStamp_eeg = [baseline_eeg(1,1), baseline_eeg(end,1), stimuli_eeg(1,1), stimuli_eeg(end,1)]; %(s)단위
    
    % time stamp 열 제거
    baseline_eeg(:,1) = []; 
    stimuli_eeg(:,1) = [];
    
    fprintf('EEG baseline size :');
    disp(size(baseline_eeg));
    fprintf('EEG stimuli size :');
    disp(size(stimuli_eeg));

    % Save csv file
    filename = char(save_path_EEG + "baseline\\s" + subject + "_" + sample + ".csv");
    writematrix(baseline_eeg, filename);
    filename = char(save_path_EEG + "stimuli\\s" + subject + "_" + sample + ".csv");
    writematrix(stimuli_eeg, filename);
    
    
    
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
    data = ECG_data{:,4:6};
               
    % Synchronize EEG, ECG time stamp
    timeStamp_ECG = [1,1,1,1];
    index = 1;
    ECG_timeIndex = [1,1,1,1];
    timeStamp = ECG_data{:,1}.';
    timeStamp = timeStamp./1000;
    previousTime = timeStamp(1);
    for time = timeStamp
        if previousTime <= timeStamp_eeg(1) && timeStamp_eeg(1) <= time
            timeStamp_ECG(1) = previousTime;
            ECG_timeIndex(1) = index;
        end
        if previousTime <= timeStamp_eeg(2) && timeStamp_eeg(2) <= time
            timeStamp_ECG(2) = previousTime;
            ECG_timeIndex(2) = index;
        end
        if previousTime <= timeStamp_eeg(3) && timeStamp_eeg(3) <= time
            timeStamp_ECG(3) = previousTime;
            ECG_timeIndex(3) = index;
        end
        if previousTime <= timeStamp_eeg(4) && timeStamp_eeg(4) <= time
            timeStamp_ECG(4) = previousTime;
            ECG_timeIndex(4) = index;
        end
        previousTime = time;
        index = index + 1;
    end

    % Data Cut-off
    data(ECG_timeIndex(2):ECG_timeIndex(3), :) = [];    
    baseline_ecg = data(ECG_timeIndex(1):ECG_timeIndex(2), :);    % baseline
    adjustIndex = ECG_timeIndex(3) - ECG_timeIndex(2) - 1;
    ECG_timeIndex(3) = ECG_timeIndex(3) - adjustIndex;
    ECG_timeIndex(4) = ECG_timeIndex(4) - adjustIndex;
    stimuli_ecg = data(ECG_timeIndex(3):ECG_timeIndex(4), :);     % stimuli
    
    
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
