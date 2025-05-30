#!/bin/bash

# fix
SEED=42
NUM_PROCS=2
BS_PER_PROC=32
CURRENT_DATE="0308"
LIBERO_SUBSUITE="libero_10"

# hyper
PORT=29581
AVAILABLE_GPUS="0,1"
MODEL_NAME="lbp_policy_ddpm_res34_libero"
RECURSIVE_PLANNING_STEP=2
EXPERIMENT_NAME="runnings/${CURRENT_DATE}-${MODEL_NAME}-rec_${RECURSIVE_PLANNING_STEP}-${LIBERO_SUBSUITE}-bs_$((NUM_PROCS*BS_PER_PROC))-seed_${SEED}"

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_PROCS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=${PORT} \
    train_policy_sim.py \
        --seed $SEED \
        --output_dir $EXPERIMENT_NAME \
        --gpus $AVAILABLE_GPUS \
        --num_iters 200000 \
        --chunk_length 6 \
        --model_name $MODEL_NAME \
        --engine_name build_libero_engine \
        --dataset_path /mnt/ssd0/data/libero/$LIBERO_SUBSUITE \
        --img_size 224 \
        --batch_size $BS_PER_PROC \
        --num_workers 8 \
        --use_ac True \
        --learning_rate 3e-4 \
        --weight_decay 0 \
        --eta_min_lr 0 \
        --save_interval 20000 \
        --warm_steps 2000 \
        --log_interval 50 \
        --recursive_step $RECURSIVE_PLANNING_STEP \
        --imaginator_ckpt_path /home/ldx/LBP/runnings/0307-mid_planner_libero_dnce-bs_64-seed_42/Model_ckpt_100000.pth
