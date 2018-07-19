contextLength = 2176;
targetLength = 768;
contextRatio = ceil(contextLength/targetLength);
maxLag = 1000;

audioFilePath = 'audio/bass_electronic_018-045-075.wav';
[audio, Fs] = audioread(audioFilePath);

t = linspace(0, pi/2, targetLength)';
sqCos = cos(t).^2;

rec_signal = [];
SNR = [];

for i = contextRatio:(length(audio)/targetLength)-contextRatio-2
    previous_sig = audio(targetLength*(i-contextRatio)+1:targetLength*(i));
    target_sig = audio(targetLength*(i)+1:targetLength*(i+1));
    next_sig = audio(targetLength*(i+1)+1:targetLength*(i+contextRatio+1));
    
    if rms(target_sig) < 1e-4
        SNR(length(SNR)+1) = -1;
        rec_signal = cat(1, rec_signal, zeros([targetLength, 1]));
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
    rec_signal = cat(1, rec_signal, sigout);
    SNR(length(SNR)+1) = mySNR(target_sig, sigout);
end


fprintf('mean SNR where it was calculated is %f \n', mean(SNR(SNR~=-1)));
fprintf('max SNR is %f \n', max(SNR));
fprintf('SNR is not calculated at %d places \n', length(find(SNR==-1)));

