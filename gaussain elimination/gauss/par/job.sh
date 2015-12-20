#!/bin/bash
exe=gauss_par
job='jpwh_991.dat'
echo '///////////////////////////////////////////////'
echo Input matrix: $job
for i in 1 2 4 8 16;do
	echo process number: $i
	./$exe -m ./input_matrices/$job -p$i
	./$exe -m ./input_matrices/$job -p$i
done

job='orsreg_1.dat'
echo '///////////////////////////////////////////////'
echo Input matrix: $job
for i in 1 2 4 8 16;do
	echo process number: $i
	./$exe -m ./input_matrices/$job -p$i
	./$exe -m ./input_matrices/$job -p$i
done
