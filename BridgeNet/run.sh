CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--master_port 17232 \
--nproc_per_node=2  \
--use_env \
main.py \
--config_exp './configs/pascal/pascal_vitL_bridgenet.yml' \
--run_mode train \
--dataset_log 'PASCAL' \
#--trained_model  pretrained_model_path
