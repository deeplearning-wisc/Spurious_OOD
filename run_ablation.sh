#!/bin/bash

for method in "erm"
do
    for epoch in 1 2 3 4 5 6
        do
        # for r in 0.5 0.7 0.9
        for r in 1.0
        do
            NOW=$( date '+%F_%H:%M:%S' )
            r_str=$( printf "%.1f" $r )
            num=$( printf $r_str | tail -c 1 )
            if [[ $r_str = "1.0" ]]
            then
                name="${method}_r_1"
            else
                name="${method}_r_0_$num"
            fi
            echo $name
            log_name="info_${epoch}_${NOW}.log"
            val_log_name="info_val_${epoch}_${NOW}.log"
            mkdir -p results_non/$name
            python train_bg.py --method $method --name $name --data_label_correlation $r --gpu-ids 2  --log_name ${log_name}
            python eval_ood_detection.py --name $name --epochs best --model-arch resnet18 --gpu 2 --data_label_correlation $r --in-dataset color_mnist --method msp
            python compute_metrics.py --name $name --in-dataset color_mnist --method msp --epochs best >> results_non/$name/result.log
        done
    done
done