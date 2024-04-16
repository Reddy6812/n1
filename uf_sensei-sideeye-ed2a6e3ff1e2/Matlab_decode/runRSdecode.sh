#!/bin/bash

testrt='/blue/srampazzi/pnaghavi/Data/'
testname='test21'
set_size=4  
num_workers=4
pybatchsize=4  
delmat=1  # set this to 1 if deleting mat file after decoding
ordercorrect=1  

for set_num in {1,2}
do
	echo "Running Python..."
	python vids2mat.py ${testrt} ${testname} ${set_size} ${set_num}  ${num_workers} ${pybatchsize}
	echo "Running Matlab..."
	/apps/matlab/r2020b/bin/matlab -nodisplay -nosplash -nodesktop -r "decodeRS('${testrt}', '${testname}',  ${set_size}, ${set_num}, ${num_workers}, 'mat', ${delmat}, ${ordercorrect}, 'DMiter', 10) ;exit;"
done
