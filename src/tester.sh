#!/bin/sh
#test_param=0
for i in `seq 1 5`;
do
    #test_param=$(($i*8))
    #echo Neurons Num Equals: $test_param
    python3 main.py $i >> results.txt
done
