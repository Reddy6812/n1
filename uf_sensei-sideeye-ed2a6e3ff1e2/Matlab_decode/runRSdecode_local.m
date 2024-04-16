testrt = 'S:\AudioMNIST\test41_scenarios';
testname = 'test_logitechspeaker_full';
set_size = 20;
% set_num = 8;
num_workers = 8;
vidsuffix = 'mp4';
delmat = 0;
ordercorrect = 1;   % we do it in the post processing in local machines

show_progress = 1;

all_sets = 1:500;


delete('log.txt')
fid = fopen('log.txt', 'w+');
fprintf(fid, [testname, '\n', num2str(all_sets(1))...
    , ' to ', num2str(all_sets(end))]);
fclose(fid);

if show_progress
    progressbar()
end
for set_num = all_sets
    decodeRS(testrt, testname, set_size, set_num, num_workers, vidsuffix, delmat, ordercorrect)
    if show_progress
        progressbar((set_num-all_sets(1)+1)/length(all_sets));
    end
end