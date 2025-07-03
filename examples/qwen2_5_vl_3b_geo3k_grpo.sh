#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH="/root/autodl-tmp/model/Qwen/Qwen2.5-VL-3B-Instruct"  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \ 
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo \
    trainer.n_gpus_per_node=2 \
    trainer.save_checkpoint_path=/root/autodl-tmp/checkpoints/easy_r1/qwen2_5_vl_3b_geo_grpo \
    trainer.total_epochs=2 \
    trainer.max_steps=5 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    > logs/log.txt

# Training Datas: hiyouga/geometry3k about Geometry