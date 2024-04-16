[ref, fs] = audioread('../original_norm_add0.wav');
[deg, ~] = audioread('../recovery_norm_add0.wav');
lendiff = length(ref) - length(deg);
if lendiff < 0
    ref = [ref; zeros(-lendiff, 1)];
else
    deg = [deg; zeros(lendiff, 1)];
end



% [ref_a,deg_a] = alignsignals(ref,deg);



disp(['Intiligibility = ', num2str(stoi(ref, deg, fs))]) 

