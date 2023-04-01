# Deep_cluster


# Upstream Training
```
CUDA_VISIBLE_DEVICES=2,3,4,5 python main.py \
    --input dummy.csv \
    --epochs 200 \
    --batch_size 512 \
    --cluster_algo kmeans \
    --save_dir "path_to_dir_to_be_saved" \
    --num_workers 2
    --resume
    --checkpoint_path /speech/srayan/AAAI/DECAR/upstream/best_loss.pth.tar
```
