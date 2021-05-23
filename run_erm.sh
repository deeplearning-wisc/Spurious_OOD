#!/bin/bash

# for method in "erm"
# do
#     for epoch in 1 2 3 4 5 6 7 8
#         do
#         for r in 0.5 0.7 0.9 1
#         # for r in 1.0
#         do
#             NOW=$( date '+%F_%H:%M:%S' )
#             r_str=$( printf "%.1f" $r )
#             num=$( printf $r_str | tail -c 1 )
#             if [[ $r_str = "1.0" ]]
#             then
#                 name="${method}_r_1"
#             else
#                 name="${method}_r_0_$num"
#             fi
#             echo $name
#             log_name="info_${epoch}_${NOW}.log"
#             val_log_name="info_val_${epoch}_${NOW}.log"
#             mkdir -p results_binary/$name
#             python train_bg.py --method $method --name $name --data_label_correlation $r --gpu-ids 4  --log_name ${log_name}
#             python test_bg.py --method $method --name $name --data_label_correlation $r --gpu-ids 4 --log_name ${val_log_name} --test_epochs best
#             python get_results.py --test_epochs best --name $name >> results_binary/$name/result.log
#         done
#     done
# done

for method in "erm"
do
    for epoch in 1 2 3 4 5 6 7 8
        do
        for r in 0.5 0.7 0.9 1.0
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
            mkdir -p results_waterbird/$name
            # --lr 0.001 --weight-decay 0.001 
            # --lr 0.0001 --weight-decay 0.001
            python train_bg.py --gpu-ids 7 --method $method --name $name --in-dataset waterbird --data_label_correlation $r \
            --model resnet50 --epochs 100 --lr 0.00001 --weight-decay 0.05 --log_name ${log_name} --save-epoch 50
            python test_bg.py --method $method --name $name  --data_label_correlation $r --gpu-ids 7 -b 64 --test_epochs best \
            --model resnet50 --in-dataset waterbird --log_name ${val_log_name}
            python get_results.py --in-dataset waterbird --test_epochs best --name $name >> results_waterbird/$name/result.log
        done
    done
done
