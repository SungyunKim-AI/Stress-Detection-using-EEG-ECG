% =========================== ECG Preprocessing ===========================
numOfSubjects = 5;
numOfSamples = 10;
samplingRate_ECG = 512;
load_path_ECG = "C:\\Users\\user\\Desktop\\data_preprocessed\\cutoff_preprocessed\\ECG\\";
save_path_ECG = "C:\\Users\\user\\Desktop\\data_preprocessed\\ECG_preprocessed\\";

% Fnotch = 60;    % Notch Frequency
% BW = 120;       % Bandwidth
% Apass = 1;      % Bandwidth Attenuation
% [b, a] = iirnotch(Fnotch/(samplingRate_ECG/2), BW/(samplingRate_ECG/2), Apass);
% Hd = dfilt.df2 (b, a);
% 
% 
% for category = ["baseline", "stimuli"]
%     for subject = 1:numOfSubjects
%         for sample = 1:numOfSamples
%             
%             % Load Data
%             file_path = char(load_path_ECG + category + "\\s" + subject + "_" + sample + ".csv");
%             ECG_data = readtable(file_path,"VariableNamingRule","preserve");
%             data = ECG_data{:,:};
% 
%             for i = 1:3
%                 [C, L] = wavedec (data(:,i), 9, 'bior3.7');     % Decomposition
%                 a9 = wrcoef ('a', C, L,'bior3.7',9);            % 'a' = Approximate Component
%                 d9 = wrcoef ('d', C, L,'bior3.7',9);            % 'd' = Detailed components
%                 d8 = wrcoef ('d', C, L,'bior3.7',8);
%                 d7 = wrcoef ('d', C, L,'bior3.7',7);
%                 d6 = wrcoef ('d', C, L,'bior3.7',6);
%                 d5 = wrcoef ('d', C, L,'bior3.7',5);
%                 d4 = wrcoef ('d', C, L,'bior3.7',4);
%                 d3 = wrcoef ('d', C, L,'bior3.7',3);
%                 d2 = wrcoef ('d', C, L,'bior3.7',2);
%                 d1 = wrcoef ('d', C, L,'bior3.7',1);
% 
%                 % ECG Signal after baseline wander REMOVED
%                 y = d9+d8+d7+d6+d5+d4+d3+d2+d1;
% 
%                 % ECG signal with power line noise Removed
%                 y1 = filter(Hd, y);
% 
%                 data(:,i) = y1;
% 
%                 subplot (3,1,1), plot(data(:,1)), title ('ECG Signal with baseline wander'), grid on
%                 subplot (3,1,2), plot(y), title ('ECG Signal after baseline wander REMOVED'), grid on
%                 subplot (3,1,3), plot (y1), title ('ECG signal with powerline noise Removed'), grid on 
%             end
%             
%             fileName = char(save_path_ECG + category + "\\s" + subject + "_" + sample + ".csv");
%             writematrix(data, fileName);
%         end
%     end
% end

file_path = char(load_path_ECG + "baseline\\s" + 1 + "_" + 1 + ".csv");
ECG_data = readtable(file_path,"VariableNamingRule","preserve");
data = ECG_data{:,:};

%  BLW removal method based on FIR Filter
%
%  ecgy:        the contamined signal
%  Fc:          cut-off frequency
%  Fs:          sample frequiency
%  ECG_Clean :  processed signal without BLW

ecgy = data(:,2);
Fc = 0.667;
Fs = 51.2;
fcuts = [(Fc-0.07) (Fc)];
mags = [0 1];
devs = [0.005 0.001];

[n,Wn,beta,ftype] = kaiserord(fcuts,mags,devs,Fs);
%n = 28;
b = fir1(n,Wn,ftype,kaiser(n+1,beta),'noscale');
a = 1;

ECG_Clean = filtfilt(b,a,ecgy);
subplot (2,1,1), plot(data(:,2)), title ('ECG Signal with baseline wander'), grid on
subplot (2,1,2), plot(ECG_Clean), title ('ECG Signal after baseline wander REMOVED'), grid on



