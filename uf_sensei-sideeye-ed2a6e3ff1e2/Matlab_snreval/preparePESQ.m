function preparePESQ()

cd('L:\ubuntu\SPQR\Projects\Camera\rolling\snreval-master\pesq-mex-master')
mex *.c -output ./bin/PESQ_MEX
addpath('./bin');
addpath('./')
cd('L:\ubuntu\SPQR\Projects\Camera\rolling\snreval-master\')