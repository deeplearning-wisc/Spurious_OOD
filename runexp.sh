#!/bin/bash
name=`date +"%Y-%m-%d_%H.%M.%S"`
data=$1
corr=$2
gpu=$3
TS=(
    0
    # 400
    # # 300
    # # 256
    200
    # # 150
    # # 128
    100
    # # 75
    # # 64
    50
)

echo $name
echo "Train"
python train_bg.py --gpu-ids $gpu --in-dataset $data --model resnet18 --epochs 30 --save-epoch 30 --data_label_correlation $corr --domain-num 4 --method erm --name $name  --lr 0.001 --weight-decay 0.001
echo "Get activations"
python get_activations.py --gpu-ids $gpu --in-dataset $data --model resnet18 --test_epochs 30 --data_label_correlation $corr --method erm --name $name  --root_dir datasets/ood_datasets 

python compare_activations.py $data $name 30

for t in "${TS[@]}"; do
    echo "TOP $t"
    echo "Test"
    python test_bg.py --gpu-ids $gpu --in-dataset $data --model resnet18 --test_epochs 30 --data_label_correlation $corr --method gdro --name $name  --root_dir datasets/ood_datasets -t $t
    echo "Present results"
    python present_results.py --in-dataset $data --name $name  --test_epochs 30 -t $t
done

python notify.py
echo $name
