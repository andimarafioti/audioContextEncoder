function [result] = mySNR(orig_signal, inpainted)

norm_orig =  norm(orig_signal);
norm_difference = norm(orig_signal-inpainted);
result = 10*log10(abs(norm_orig^2)/(abs(norm_difference^2)));
end