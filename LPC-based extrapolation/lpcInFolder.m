contextLength = 2048;
targetLength = 1024;
contextRatio = ceil(contextLength/targetLength);
maxLag = 1000;

folder = 'fma';
extension = 'mp3';
audiofiles = dir(strcat(folder, '/*', extension));
allSNR = [];

for file = audiofiles'

fprintf(1,'Inpainting %s\n', file.name)
[audio, Fs]=audioread(strcat(folder, '/', file.name));

t = linspace(0, pi/2, targetLength)';
sqCos = cos(t).^2;

SNR = [];

for i = contextRatio:(length(audio)/targetLength)-contextRatio-2
    previous_sig = audio(targetLength*(i-contextRatio)+1:targetLength*(i));
    target_sig = audio(targetLength*(i)+1:targetLength*(i+1));
    next_sig = audio(targetLength*(i+1)+1:targetLength*(i+contextRatio+1));
    
    if rms(target_sig) < 1e-4
        continue
    end  

    ab = arburg(previous_sig, maxLag);
    Zb = filtic(1,ab,previous_sig(end-(0:(maxLag-1))));
    forw_pred = filter(1,ab,zeros(1,targetLength),Zb)';

    next_sig = flipud(next_sig);
    af = arburg(next_sig, maxLag);
    Zf = filtic(1,af, next_sig(end-(0:(maxLag-1))));
    backw_pred = flipud(filter(1,af,zeros(1,targetLength),Zf)');
    
    sigout = sqCos.*forw_pred + flipud(sqCos).*backw_pred;
    SNR(length(SNR)+1) = mySNR(target_sig, sigout);
end

fprintf('mean SNR is %f \n', mean(SNR));

allSNR = cat(2, SNR, allSNR);

end

allSNR(isnan(allSNR)) = 0;

fprintf('mean SNR is %f \n', mean(allSNR));
fprintf('std SNR is %f \n', std(allSNR));
fprintf('min SNR is %f \n', min(allSNR));
fprintf('25%% percentile SNR is %f \n', prctile(allSNR, 25));
fprintf('50%% percentile SNR is %f \n', prctile(allSNR, 50));
fprintf('75%% percentile SNR is %f \n', prctile(allSNR, 75));
fprintf('max SNR is %f \n', max(SNR));
