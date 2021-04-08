% ========================== Cut-off Data =================================
% baseline : 눈 뜨고 15초, 눈 감고 15초 총 40초 잘라내기 → 대략 1분 20초
% stimuli : 앞 부분 5초, 뒷부분 2초 제거 → 대략 1분 8초 남는다.
% experimentTime : 실험 시간 (초 단위)
% restEnd : 휴식 종료 시간 (초 단위)
% cptStart : CPT 시작 시간 (초 단위)

clear;
clc;

% => subject1 : 김찬우
% subject = 1;
% experimentTime = [195, 190, 196, 195, 210, 195, 200, 195, 195, 205];
% restEnd =        [123, 121, 121, 120, 134, 120, 125, 120, 120, 130];
% cptStart =       [133, 128, 123, 120, 135, 123, 125, 130, 120, 137];

% => subject2 : 조윤혁
% subject = 2;
% experimentTime = [195, 195, 195, 195, 195, 195, 195, 195, 195, 195];
% restEnd =        [120, 120, 120, 120, 120, 117, 120, 120, 120, 120];
% cptStart =       [120, 125, 126, 120, 120, 126, 125, 126, 122, 120];

% => subject3 : 김인애
% subject = 3;
% experimentTime = [195, 195, 195, 195, 195, 195, 195, 195, 195, 195];
% restEnd =        [120, 120, 120, 120, 120, 120, 120, 120, 120, 120];
% cptStart =       [125, 124, 126, 124, 125, 125, 126, 125, 124, 124];

% => subject4 : 안이솔
% subject = 4;
% experimentTime = [195, 195, 195, 195, 193, 210, 195, 195, 196, 196];
% restEnd =        [120, 120, 120, 120, 120, 135, 120, 120, 121, 121];
% cptStart =       [122, 123, 122, 120, 120, 137, 122, 122, 121, 124];

% => subject5 : 김성윤
% subject = 5;
% experimentTime = [195, 195, 195, 195, 193, 195, 195, 195, 195, 195];
% restEnd =        [120, 120, 120, 120, 120, 122, 121, 122, 120, 120];
% cptStart =       [123, 123, 123, 123, 123, 128, 122, 124, 123, 122];

% => subject6 : 왕윤성
% subject = 6;
% experimentTime = [195, 195, 195, 195, 195, 195, 195, 195, 195, 195];
% restEnd =        [120, 120, 120, 122, 120, 120, 120, 121, 120, 120];
% cptStart =       [127, 124, 123, 125, 123, 123, 125, 124, 123, 124];

% => subject7 : 유종우
% subject = 7;
% experimentTime = [195, 195, 195, 195, 199, 195, 195, 195, 196, 197];
% restEnd =        [120, 120, 121, 121, 125, 121, 123, 120, 120, 122];
% cptStart =       [126, 125, 127, 128, 132, 127, 125, 126, 125, 127];

% => subject8 : 강범석
% subject = 8;
% experimentTime = [195, 195, 195, 195, 195, 195, 195, 195, 195, 195];
% restEnd =        [121, 121, 121, 120, 121, 121, 119, 114, 122, 120];
% cptStart =       [125, 124, 125, 125, 127, 125, 126, 125, 129, 126];

% => subject9 : 김찬우
% subject = 9;
% experimentTime = [190, 195, 195, 195, 195, 195, 195, 195, 200, 195];
% restEnd =        [120, 120, 120, 120, 120, 120, 120, 120, 125, 120];
% cptStart =       [124, 125, 126, 125, 123, 125, 125, 123, 132, 125];

noOfSamples = 10;
SamplingRate_EEG = 128;      % Emotive EpocX Sampling Rate (Hz 단위)

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
    EEG_data = readtable(file_path,"VariableNamingRule","preserve");

    % Data Cut-off
    experimentTime(sample) = experimentTime(sample) * SamplingRate_EEG;
    exData = EEG_data{1:experimentTime(sample), [1,4:17]};  % table 데이터를 matrix 데이터로 변경  
    cptStart(sample) = cptStart(sample) * SamplingRate_EEG;
    restEnd(sample) = restEnd(sample) * SamplingRate_EEG;
    stimuli_eeg = exData(cptStart(sample):end, :);          % baseline
    baseline_eeg = exData(1:restEnd(sample),:);             % stimuli

    baseline_eeg(1:5120, :) = [];                % 앞 부분 40초 제거
    stimuli_eeg([1:256, end-256:end], :) = [];   % 앞 부분 5초, 뒷 부분 2초 제거
    timeStamp_EEG = [baseline_eeg(1,1), baseline_eeg(end,1), stimuli_eeg(1,1), stimuli_eeg(end,1)]; %(s)단위
    
    baseline_eeg(:,1) = []; % time stamp 열 제거
    stimuli_eeg(:,1) = [];  % time stamp 열 제거
    
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
    if subject < 5
        continue;
    end
    
    fprintf('=> ECG\n');
    % Data load
    file_path = char(load_path_ECG + "s" + subject + "_" + sample + ".csv");
    ECG_data = readtable(file_path,"VariableNamingRule","preserve");
    data = ECG_data{:,4:6};
    
    % Bandpass filter (1 ~ 50Hz)
    Fs = 512;
    fcuts = [0.5 1.0 45 46];
    mags = [0 1 0];
    devs = [0.05 0.01 0.05];
    for i = 1:3   
        [n,Wn,beta,ftype] = kaiserord(fcuts,mags,devs,Fs);
        n = n + rem(n,2);
        hh = fir1(n,Wn,ftype,kaiser(n+1,beta),'noscale');
        data(:,i) = filtfilt(hh,1,data(:,i));
    end
               
    % Synchronize EEG, ECG time stamp
    timeStamp_ECG = [1,1,1,1];
    index = 1;
    ECG_timeIndex = [1,1,1,1];
    timeStamp = ECG_data{:,1}.';
    timeStamp = timeStamp./1000;
    previousTime = timeStamp(1);
    for time = timeStamp
        if previousTime <= timeStamp_EEG(1) && timeStamp_EEG(1) <= time
            timeStamp_ECG(1) = previousTime;
            ECG_timeIndex(1) = index;
        end
        if previousTime <= timeStamp_EEG(2) && timeStamp_EEG(2) <= time
            timeStamp_ECG(2) = previousTime;
            ECG_timeIndex(2) = index;
        end
        if previousTime <= timeStamp_EEG(3) && timeStamp_EEG(3) <= time
            timeStamp_ECG(3) = previousTime;
            ECG_timeIndex(3) = index;
        end
        if previousTime <= timeStamp_EEG(4) && timeStamp_EEG(4) <= time
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
