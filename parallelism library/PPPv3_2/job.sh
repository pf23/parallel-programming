#!/bin/bash
exe=driver-compute
echo $exe
echo '///////////////////////////////////////////////'
for grainsize in  40000000 10000000 5000000 1000000 500000 100000\
					50000 10000 5000 1000 500 100 50 20;do
	echo grainsize: $grainsize
	./$exe -p 40000000 -t 5 -g $grainsize -n 4
done
