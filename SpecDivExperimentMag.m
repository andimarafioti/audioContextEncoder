
%% STFT parameters 

win = {'hann',512,'peak'};
dual = {'dual',win};
M = 512; a = M/4;
flag = 'timeinv';
gamma = pghi_findgamma(win);

%% Obtain data - THIS MUST BE UPDATED ONCE STUFF IS AVAILABLE!


% load('magnitude_trainedOnFma_step723261_8mslater.mat')
% tfdata_amp = magnitudeMat;
% clear magnitudeMat;
% load('magnitude_trainedOnFma_step723261_8msbefore.mat')
% t = linspace(0, pi/2, 7)';
% sqCos = permute(repmat(cos(t).^2, 1, 3328, 257), [2 1 3]);
% tfdata_amp = [magnitudeMat(:, 3:4, :) (tfdata_amp(:, 1:end-4, :).*sqCos+fliplr(sqCos).*magnitudeMat(:, 5:end, :)) tfdata_amp(:, end-3:end-2, :)];
% clear magnitudeMat;
% 

load('magnitude_trainedOnFma_step723261_8mslater.mat')
later = magnitudeMat;
clear magnitudeMat;
load('magnitude_trainedOnFma_step723261_8msbefore.mat')
before = magnitudeMat;
clear magnitudeMat;
load('magnitude_trainedOnFma_step723261.mat')
central = magnitudeMat;
clear magnitudeMat;

tfdata_amp = [(before(:, 3:4, :)+central(:, 1:2, :))/2 (later(:, 1:end-4, :)+central(:, 3:end-2,:)+before(:, 5:end, :))/3 (central(:, end-1:end,:)+later(:, end-3:end-2, :))/2];
% tfdata_amp = central;

load('FMA_test_windows_16k.mat'); 
alldata_ori = fma_test(1:length(tfdata_amp),5121:5120*2).';
clear fma_test;

load('CE_FMAonly_step2547124.mat'); 
alldata_rim = CEMat(1:length(tfdata_amp),:).';
clear generatedTimeSignals;

load('fma_lpcrec_16k.mat')
alldata_lpc = out(:,:).';
clear out;

%num_data = 10;
num_data = size(alldata_ori,2);
num_methods = 10;

L = 5120;

num_tframes = 40; 
num_unknown = 11; 

%% Prepare arrays for results

SpecDiv = zeros(num_data,num_methods);
SNR = zeros(num_data,num_methods);

mask = zeros(M/2+1,L/a);
mask(:,[1:15,end-13:end]) = 1;

known_idx = [1:15,num_tframes-13:num_tframes];
idx = 19:(num_tframes-17);
idx = 20:(num_tframes-18); % for 48ms
%idx = 1:40;

known_tidx = [1:(14*a),(L-12*a)+1:L];
tidx = (16*a)+1:(L-16*a);
%tidx = 1:L;

%% Compute error measures

for kk = 1:num_data    
    % Load waveform data
    data_ori = alldata_ori(:,kk);
    data_ori_nomean = data_ori;% - mean(data_ori);
    data_lpc = alldata_lpc(:,kk);
    data_rim = alldata_rim(:,kk);
   
    c_ori = dgtreal(data_ori,win,a,M,L,flag); % DGT of original
    c_angle_ori = angle(c_ori); % Phase of original DGT

    % LPC
%     SNR(kk,1) = 20*log10(norm(data_ori_nomean(tidx))/norm(data_lpc(tidx)-data_ori(tidx)));
%     c_lpc = dgtreal(data_lpc,win,a,M,L,flag);
%     SpecDiv(kk,1) = 20*log10(1/magnitudeerr(c_ori(:,idx),c_lpc(:,idx)));   
     
    % Real and Imag
%     SNR(kk,2) = 20*log10(norm(data_ori_nomean(tidx))/norm(data_rim(tidx)-data_ori(tidx)));
%     c_rim = dgtreal(data_rim,win,a,M,L,flag);
%     SpecDiv(kk,2) = 20*log10(1/magnitudeerr(c_ori(:,idx),c_rim(:,idx)));  
%     % Real and Imag + Original Phase
%     c_rim_tp = abs(c_rim).*exp(1i*c_angle_ori);
%     f_rim_tp = idgtreal(c_rim_tp,dual,a,M,flag);
%     SNR(kk,3) = 20*log10(norm(data_ori_nomean(tidx))/norm(f_rim_tp(tidx)-data_ori(tidx)));
%     c_rim_rec = dgtreal(f_rim_tp,win,a,M,L,flag);
%     SpecDiv(kk,3) = 20*log10(1/magnitudeerr(c_ori(:,idx),c_rim_rec(:,idx)));  
%     % Real and Imag + PGHI
%     c_rim_pghi = pghi(c_rim,pghi_findgamma(win),a,M,mask,flag);
%     f_rim_pghi = idgtreal(c_rim_pghi,dual,a,M,flag);
%     SNR(kk,4) = 20*log10(norm(data_ori_nomean(tidx))/norm(f_rim_pghi(tidx)-data_ori(tidx)));
%     c_rim_pghi_rec = dgtreal(f_rim_pghi,win,a,M,L,flag);
%     SpecDiv(kk,4) = 20*log10(1/magnitudeerr(c_ori(:,idx),c_rim_pghi_rec(:,idx))); 
%     % Real and Imag + FGLIM
    c_rim_gla = masked_gla(c_rim,dual,a,M,mask,flag,'fgla','input');
    f_rim_gla = idgtreal(c_rim_gla,dual,a,M,flag);
    SNR(kk,5) = 20*log10(norm(data_ori_nomean(tidx))/norm(f_rim_gla(tidx)-data_ori(tidx)));
    c_rim_gla_rec = dgtreal(f_rim_gla,win,a,M,L,flag);
    SpecDiv(kk,5) = 20*log10(1/magnitudeerr(c_ori(:,idx),c_rim_gla_rec(:,idx))); 
%     % Real and Imag + PGHI + FGLIM
%     c_rim_pgla = gla(c_rim_pghi,dual,a,M,flag,'fgla','input');
%     SpecDiv(kk,6) = 20*log10(1/magnitudeerr(c_ori(:,idx),c_rim_pgla(:,idx))); 
%     f_rim_pgla = idgtreal(c_rim_pgla,dual,a,M,flag);
%     SNR(kk,6) = 20*log10(norm(data_ori_nomean(tidx))/norm(f_rim_pgla(tidx)-data_ori(tidx)));
%     c_rim_pgla_rec = dgtreal(f_rim_pgla,win,a,M,L,flag);
%     SpecDiv(kk,6) = 20*log10(1/magnitudeerr(c_ori(:,idx),c_rim_pgla_rec(:,idx))); 
    
    % Amplitude (original phase)
    c_amp = abs(c_ori); % Initialize magnitude
    c_amp(:,16:(num_tframes-14)) = squeeze(tfdata_amp(kk,:,:)).'; % Set inner part to proposed solution
    c_amp_tp = abs(c_amp).*exp(1i*c_angle_ori);
    f_amp_tp = idgtreal(c_amp_tp,dual,a,M,flag);
    SNR(kk,7) = 20*log10(norm(data_ori_nomean(tidx))/norm(f_amp_tp(tidx)-data_ori(tidx)));
    c_amp_rec = dgtreal(f_amp_tp,win,a,M,L,flag);
    SpecDiv(kk,7) = 20*log10(1/magnitudeerr(c_ori(:,idx),c_amp_rec(:,idx)));  
    % Amplitude + PGHI
    kphase = (c_angle_ori.*mask);%+2*pi*rand(M/2+1,num_tframes).*(1-mask));
    c_amp_kphase = c_amp.*exp(1i*kphase);
    c_amp_pghi = pghi(c_amp_kphase,gamma,a,M,mask,flag);
    f_amp_pghi = idgtreal(c_amp_pghi,dual,a,M,flag);
    SNR(kk,8) = 20*log10(norm(data_ori_nomean(tidx))/norm(f_amp_pghi(tidx)-data_ori(tidx)));
    c_amp_pghi_rec = dgtreal(f_amp_pghi,win,a,M,L,flag);
    SpecDiv(kk,8) = 20*log10(1/magnitudeerr(c_ori(:,idx),c_amp_pghi_rec(:,idx))); 
    % Amplitude + FGLIM
%     c_amp_gla = masked_gla(c_amp_kphase,dual,a,M, mask,flag,'fgla','input');
%     f_amp_gla = idgtreal(c_amp_gla,dual,a,M,flag);
%     SNR(kk,9) = 20*log10(norm(data_ori_nomean(tidx))/norm(f_amp_gla(tidx)-data_ori(tidx)));
%     c_amp_gla_rec = dgtreal(f_amp_gla,win,a,M,L,flag);
%     SpecDiv(kk,9) = 20*log10(1/magnitudeerr(c_ori(:,idx),c_amp_gla_rec(:,idx))); 
    % Amplitude + PGHI + FGLIM
    c_amp_pgla = masked_gla(c_amp_pghi,dual,a,M, mask,flag,'fgla','input');
    f_amp_pgla = idgtreal(c_amp_pgla,dual,a,M,flag);
    SNR(kk,10) = 20*log10(norm(data_ori_nomean(tidx))/norm(f_amp_pgla(tidx)-data_ori(tidx)));
    c_amp_pgla_rec = dgtreal(f_amp_pgla,win,a,M,L,flag);
    SpecDiv(kk,10) = 20*log10(1/magnitudeerr(c_ori(:,idx),c_amp_pgla_rec(:,idx)));     
    
    if mod(kk,200) == 0
        fprintf('-Iteration %d-',kk);
    end
end

maxSNR = max(SNR);
stdSNR = std(SNR);
minSNR = min(SNR);
meanSNR = mean(SNR);
medianSNR = median(SNR);
quant25SNR = quantile(SNR,0.25);
quant75SNR = quantile(SNR,0.75);

maxSpecDiv= max(SpecDiv);
stdSpecDiv = std(SpecDiv);
minSpecDiv = min(SpecDiv);
meanSpecDiv = mean(SpecDiv);
medianSpecDiv = median(SpecDiv);
quant25SpecDiv = quantile(SpecDiv,0.25);
quant75SpecDiv = quantile(SpecDiv,0.75);

save('MethodComparison.mat','meanSNR','minSNR','quant25SNR','medianSNR',...
    'quant75SNR','maxSNR','meanSpecDiv','minSpecDiv','quant25SpecDiv',...
    'medianSpecDiv','quant75SpecDiv','maxSpecDiv');

SNRstatsFMA = [meanSNR;stdSNR;minSNR;quant25SNR;medianSNR;quant75SNR;maxSNR];
SpecDivstatsFMA = [meanSpecDiv;stdSpecDiv;minSpecDiv;quant25SpecDiv;medianSpecDiv;quant75SpecDiv;maxSpecDiv];

save('StatsFMA.mat','SNRstatsFMA','SpecDivstatsFMA');
