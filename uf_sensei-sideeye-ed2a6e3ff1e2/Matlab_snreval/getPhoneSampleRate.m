function sr = getPhoneSampleRate(phone)
% for the audio metrics eval 
% true sample rates of the wav files (with a nominal fs=30k)
% are inferred by comparing with px2 with  px2-A0-58-9-49.wav

    phone = lower(phone);
    if contains(phone, 'px1')
        sr = 45000;
        return;
    elseif contains(phone, 'px2')
        sr = 34000;
        return;
    elseif contains(phone, 'px3')
        sr = 34000;
        return;
    elseif contains(phone, 'px5')
        sr = 30000;   % resampled from 58k
        return;
    elseif contains(phone, 's7')
        sr = 45000;
        return;
    elseif contains(phone, 's8')
        sr = 45000;
        return;
    elseif contains(phone, 's20')
        sr = 30000;    % resampled from 58k
        return;
    elseif contains(phone, 'ip7')
        sr = 30000;    % resampled from 92k
        return;
    elseif contains(phone, 'ip8')
        sr = 30000;    % resampled from 92k
        return;
    elseif contains(phone, 'ip12')
        sr = 30000;    % resampled from 92k
        return;
    else
        disp(['Cannot find this phone sample rate: ',phone])
        disp('So using px2 as default')
        sr = 34000;    % resampled from 92k
        return;
    end
end