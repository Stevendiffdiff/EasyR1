#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH="/root/autodl-tmp/model/Qwen/Qwen2.5-VL-3B-Instruct"  # 替换为你的本地模型路径

python3 -m verl.trainer.SFTmain \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.answer_key=problem \
    data.rollout_batch_size=4 \
    data.val_batch_size=2 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_sft_llava \
    trainer.n_gpus_per_node=2 \
    trainer.save_checkpoint_path=/root/autodl-tmp/checkpoints/easy_r1/qwen2_5_vl_3b_sft_llava \
    trainer.total_epochs=2 \
    trainer.max_steps=5 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    > logs/sft_log.txt