# CUDA_VISIBLE_DEVICES=1 python -u run_VPCL_pretrain.py --data pretrain_LM_3layer > print_out/pretrain.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=3 python -u run_finetune.py --data finetune_mask_l2_0dot35_epoch150_unsup_3layer > print_out/finetune.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python -u /home/gslu/tableGP_yc/dataset_openml.py > print_out/aup.txt 2>&1 &


# CUDA_VISIBLE_DEVICES=0 python -u run_sup_CC18.py --data sup_CC18_baseline > print_out/sup.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -u run_finetune_CC18.py --data best_mask_model > print_out/finetune-6.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python -u run_fewshot.py --data epoch100_acc_fewshot_5 > print_out/sup.txt 2>&1 &






# CUDA_VISIBLE_DEVICES=3 python -u run_scratch.py --data 3layer_drop_freeze_full > print_out/sup.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 python -u run_finetune.py --data dot7 > print_out/finetune-dot7.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python -u run_fewshot.py --data epoch100_acc_fewshot_5 > print_out/fewshot.txt 2>&1 &

# deepspeed --master_port 29400 --num_gpus=4 run_mask_pretrain_ds.py --deepspeed_config ds_config_yc.json > print_out/pretrain.txt 2>&1 &

