
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py \
--script ctvlt --config baseline \
--save_dir ./ \
--mode multiple --nproc_per_node 4 \
--use_wandb 0