% =============== Preprocessing Experiment Data =====================
% [공통 과정]
% 1. Hamming sinc linear phase FIR filters 적용
% 2. Remove DC offset
% 3. shift by the filter’s group delay
% [선택 과정]
% 4. CAR (Common Average Reference)
% 5. ASR (Artifact Subspace Reconstruction) + CAR
% 6. CAR + FFT
% 7. ASR + CAR + FFT