% =============== 데이터 자르기 =====================
% baseline : 눈 뜨고 15초, 눈 감고 15초 총 40초 잘라내기 → 대략 1분 20초
% stimuli : 앞 부분 5초, 뒷부분 2초 제거 → 대략 1분 8초 남는다.

% Data load
file_path_EEG = "C:\\Users\\user\\Desktop\\Experiment Data\\EEG\\";
file_path_ECG = "C:\\Users\\user\\Desktop\\Experiment Data\\ECG\\";

sub = 1;
sample = 1;
file_path = char(file_path_EEG + "s" + sub + "_" + sample + ".csv");
data = readtable(file_path);
%data = data{:,:};
data(:,1:3) = [];
data(:,15:47) = [];

startPoint = 123;   % CPT 시작 시간 입력 (초 단위)
startPoint = startPoint * 128;
baseline = data{1:startPoint,:};
stimuli = 
