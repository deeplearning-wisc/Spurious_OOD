#!/bin/bash

for epoch in 1 2 3 4 5
    do
    for r in 0.2 0.4 0.6 0.8 1.0 
    # for r in 1.0
    do
        r_str=$( printf "%.1f" $r )
        num=$( printf $r_str | tail -c 1 )
        if [[ $r_str = "1.0" ]]
        then
            name="nnew_two_dann_1"
        else
            name="nnew_two_dann_$num"
        fi
        echo $name
        mkdir -p results/$name
        python train_bg.py --model-arch dann --name $name  --data_label_correlation $r --gpu-ids 7
        python test_bg.py --model-arch dann --name $name  --data_label_correlation $r --gpu-ids 7
        python get_results.py --name $name >> results/$name/result.log
    done
done
