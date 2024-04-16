function nists = wavnists(fpath)

    [data, fs] = audioread(fpath);
    nists = zeros(size(data, 2), 1);
    data = resample(data, 16000, fs);
    fs = 16000;
    for i = 1:8
        nists(i) = nist_stnr(data(:, i), fs, 0);
    end
end