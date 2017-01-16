#!/bin/sh
#test_param=0
for i in `seq 1 9`;
do
    test_param=$((0.5+($i*0.05)))
    #echo Neurons Num Equals: $test_param
    python3 main.py $test_param
done
