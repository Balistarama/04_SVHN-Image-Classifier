#!/bin/sh
#test_param=0
for i in `seq 1 5`;
do
    for j in `seq 1 5`;
    do
        BATCH_SIZE=$((64+($i*64)))
        LAYER_2_FC_NEURONS=$((128+($j*64)))
        #echo Neurons Num Equals: $test_param
        python3 main.py $BATCH_SIZE $LAYER_2_FC_NEURONS >> results.txt
    done
done
