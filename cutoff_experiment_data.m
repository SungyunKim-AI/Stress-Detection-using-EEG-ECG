% ========================== Cut-off Data =================================
% baseline : 눈 뜨고 15초, 눈 감고 15초 총 40초 잘라내기 → 대략 1분 20초
% stimuli : 앞 부분 5초, 뒷부분 2초 제거 → 대략 1분 8초 남는다.
% experimentTime : 실험 시간 (초 단위)
% cptStart : CPT 시작 시간 (초 단위)

sub = 1;                    % 실험 대상자 수
noOfSamples = 10;
% subject1 : 찬우
experimentTime = [195, 195, 196, 195, 210, 195, 200, 195, 195, 205];
cptStart = [123, 121, 121, 120, 135, 120, 125, 120, 120, 130];        
EEG_SamplingRate = 128;     % Emotive EpocX Sampling Rate (Hz 단위)
ECG_SamplingRage = 51.2;     % Shimmer 3 Sampline Rate (Hz 단위)

load_path_EEG = "C:\\Users\\user\\Desktop\\Experiment Data\\EEG\\";
load_path_ECG = "C:\\Users\\user\\Desktop\\Experiment Data\\ECG\\";
save_path_EEG = "C:\\Users\\user\\Desktop\\data_preprocessed\\EEG\\";
save_path_ECG = "C:\\Users\\user\\Desktop\\data_preprocessed\\ECG\\";
% =========================================================================



for i = 1:noOfSamples
    fprintf('========= Sample %d of Subject %d =========\n',i,sub);
    fprintf('Total Experiment Time %d (s)\n', experimentTime(i));
    fprintf('CPT Start Time %d (s)\n', cptStart(i));
    fprintf('CPT Time %d (s)\n', experimentTime(i) - cptStart(i));
    
    % ========================= EEG ==================================
    fprintf('=> EEG\n');
    % Data load
    file_path = char(load_path_EEG + "s" + sub + "_" + i + ".csv");
    EEG_data = readtable(file_path,"VariableNamingRule","preserve");

    % Data Cut-off
    experimentTime(i) = experimentTime(i) * EEG_SamplingRate;
    exData = EEG_data{1:experimentTime(i), [1,4:17]};  % table 데이터를 matrix 데이터로 변경  
    cptStart(i) = cptStart(i) * EEG_SamplingRate;
    stimuli_eeg = exData(cptStart(i):end, :);        % baseline
    baseline_eeg = exData(1:cptStart(i),:);          % stimuli
    
    if sub == 1 && i == 5
        baseline_eeg(end-640:end, :) = [];
    end

    baseline_eeg(1:5120, :) = [];                % 앞 부분 40초 제거
    stimuli_eeg([1:640, end-256:end], :) = [];   % 앞 부분 5초, 뒷 부분 2초 제거
    timeStamp = [baseline_eeg(1,1), baseline_eeg(end,1), stimuli_eeg(1,1), stimuli_eeg(end,1)]; %(s)단위
    
    baseline_eeg(:,1) = []; % time stamp 열 제거
    stimuli_eeg(:,1) = [];  % time stamp 열 제거
    
    fprintf('EEG baseline size :');
    disp(size(baseline_eeg));
    fprintf('EEG stimuli size :');
    disp(size(stimuli_eeg));

    % Save csv file
    filename = char(save_path_EEG + "baseline\\s" + sub + "_" + i + ".csv");
    writematrix(baseline_eeg, filename);
    filename = char(save_path_EEG + "stimuli\\s" + sub + "_" + i + ".csv");
    writematrix(stimuli_eeg, filename);
    
    
    
    % ========================= ECG ==================================
    fprintf('=> ECG\n');
    % Data load
    file_path = char(load_path_ECG + "s" + sub + "_" + i + ".csv");
    ECG_data = readtable(file_path,"VariableNamingRule","preserve");
    
    % Synchronize EEG, ECG time stamp
    timeStamp2 = [1,1,1,1];
    index = 1;
    ECG_timeIndex = [1,1,1,1];
    ECG_timeStamp = ECG_data{:,1}.';
    ECG_timeStamp = ECG_timeStamp./1000;
    previousTime = ECG_timeStamp(1);
    for time = ECG_timeStamp
        if previousTime <= timeStamp(1) && timeStamp(1) <= time
            timeStamp2(1) = previousTime;
            ECG_timeIndex(1) = index;
        end
        if previousTime <= timeStamp(2) && timeStamp(2) <= time
            timeStamp2(2) = previousTime;
            ECG_timeIndex(2) = index;
        end
        if previousTime <= timeStamp(3) && timeStamp(3) <= time
            timeStamp2(3) = previousTime;
            ECG_timeIndex(3) = index;
        end
        if previousTime <= timeStamp(4) && timeStamp(4) <= time
            timeStamp2(4) = previousTime;
            ECG_timeIndex(4) = index;
        end
        previousTime = time;
        index = index + 1;
    end

    % Data Cut-off
    baseline_ecg = ECG_data{ECG_timeIndex(1):ECG_timeIndex(2), 4:6};    % baseline
    stimuli_ecg = ECG_data{ECG_timeIndex(3):ECG_timeIndex(4), 4:6};      % stimuli
    
    fprintf('ECG baseline size :');
    disp(size(baseline_ecg));
    fprintf('ECG stimuli size :');
    disp(size(stimuli_ecg));

    % Save csv file
    filename = char(save_path_ECG + "baseline\\s" + sub + "_" + i + ".csv");
    writematrix(baseline_ecg, filename);
    filename = char(save_path_ECG + "stimuli\\s" + sub + "_" + i + ".csv");
    writematrix(stimuli_ecg, filename);
end