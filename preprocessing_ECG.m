% =========================== ECG Preprocessing ===========================
% 1. Bandpass filter (1 ~ 50Hz)
% 2. Detect the R peaks of the QRS complex [24]
%    -  visual inspection was performed to correct the false acceptance and false rejection of R peaks.
% 3. Remove the abnormal RR Interval (RRI) [10]
% 4. NN (normal-to-normal) interval : Zero-one tranformation (R peaks = 1, others = 0)
% 5. Lomb Periodogram (0.04 to 20 Hz was adopted)


Fs = 512;
fcuts = [0.5 1.0 45 46];
mags = [0 1 0];
devs = [0.05 0.01 0.05];

% load_path_ECG = "C:\\Users\\user\\Desktop\\Experiment Data\\ECG";
% save_path_ECG = "C:\\Users\\user\\Desktop\\Experiment Data\\ECG_Clean";
load_path_ECG = "C:\\Users\\sungy\\Desktop\\Experiment Data\\ECG";
save_path_ECG = "C:\\Users\\sungy\\Desktop\\Experiment Data\\ECG_Clean";

 for subject = 6:10

    for sample = 1:10
        file_path = char(load_path_ECG + "\\s" + subject + "_" + sample + ".csv");
        dataTable = readtable(file_path,"VariableNamingRule","preserve");
        data = dataTable{:,:};
        
        for i = 4:6
            
            % 1. Bandpass filter (1 ~ 50Hz)
            [n,Wn,beta,ftype] = kaiserord(fcuts,mags,devs,Fs);
            n = n + rem(n,2);
            hh = fir1(n,Wn,ftype,kaiser(n+1,beta),'noscale');
            filtedData = filtfilt(hh,1,data(:,i));
            
            % 2. Detect the R peaks of the QRS complex
            % 함수 출처 : https://github.com/danielwedekind/qrsdetector/tree/d0efea0d883ea329b1110d3fa51802458d71f3b1
            [qrs_pos,filt_data,int_data,thF1,thI1] = pantompkins_qrs(filtedData, Fs);
            
            % 3. Remove the abnormal RRI
            
            % 4. Zero-one tranformation
            [row, col] = size(qrs_pos);
            qrspeaks = zeros(row, col);
            previousIndex = 1;
            for index = 1:col
                filt_data(previousIndex:qrs_pos(index)-1) = 0;
                qrspeaks(index) = filt_data(qrs_pos(index));
                filt_data(qrs_pos(index)) = 1;
                previousIndex = qrs_pos(index) + 1;
            end
            filt_data(previousIndex:end) = 0;
            
            % 5. Lomb Periodogram (0.04 to 20 Hz was adopted)
            [pxx,f] = plomb(qrspeaks,qrs_pos);
            plot(f,pxx);
            xlabel("Frequency");
            ylabel("Power");
            title("Lomb Periodogram");
            
        end
        
        fileName = char(save_path_ECG + "\\s" + subject + "_" + sample + ".csv");
        writematrix(data, fileName);
    end
 end
 

% wt = modwt(filtedData,5);
% wtrec = zeros(size(wt));
% wtrec(4:5,:) = wt(4:5,:);
% y = imodwt(wtrec,'sym4');
% 
% y = abs(y).^2;
% [qrspeaks,locs] = findpeaks(y);
% subplot(2,1,1);
% plot(y);
% hold on;
% plot(locs,qrspeaks,'o')
% xlabel("time");
% ylabel("Amplitude");
% title('R Peaks Localized by Wavelet Transform with Automatic Annotations');
%  
% [pxx,f] = plomb(qrspeaks,locs);
% subplot(2,1,2);
% plot(f,pxx);
% xlabel("Frequency");
% ylabel("Power");
% title("Lomb Periodogram");



