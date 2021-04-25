#!/bin/bash

for r in 0.2 0.4 0.6 0.8 1.0 
do
    r_str=$( printf "%.1f" $r )
    num=$( printf $r_str | tail -c 1 )
    if [[ $r_str = "1.0" ]]
    then
        name="rebias_1"
    else
        name="rebias_$num"
    fi
    echo $name
    # mkdir -p results/$name
    # python train_bg.py --model-arch rebias_conv --name $name  --data_label_correlation $r --gpu-ids 0
    python test_bg.py --model-arch rebias_conv --name $name  --data_label_correlation $r --gpu-ids 0
    # python get_results.py --name $name > results/$name/result.log
done