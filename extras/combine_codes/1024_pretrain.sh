#! /bin/bash
eval "$(conda shell.bash hook)"
conda activate moco_work
python /nlsasfs/home/nltm-pilot/ashishs/DeloresM/combine_codes/train_moco.py \
    --input /nlsasfs/home/nltm-pilot/ashishs/pretrain_shuffled_short.csv \
    --batch-size 10 \
    --model_type slicer




