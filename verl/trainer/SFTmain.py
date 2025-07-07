# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import sys
import torch

print("当前工作目录（os.getcwd()）:", os.getcwd())
print("sys.path:", sys.path)
print("当前文件绝对路径:", os.path.abspath(__file__))
print("当前文件所在目录:", os.path.dirname(os.path.abspath(__file__)))
print("根目录下所有文件和文件夹:", os.listdir(os.getcwd()))

root_dir = os.getcwd()
print("根目录下所有包（含 __init__.py 的文件夹）:")
for name in os.listdir(root_dir):
    full_path = os.path.join(root_dir, name)
    if os.path.isdir(full_path) and "__init__.py" in os.listdir(full_path):
        print("  -", name)

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_SFTTrainer import RaySFTTrainer, ResourcePoolManager, Role
from ..single_controller.base import Worker

# 可选：自定义loss函数示例
def custom_loss(outputs, labels):
    # 用户可自定义loss逻辑，否则用默认label smoothing
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

@ray.remote(num_cpus=1)
class Runner:
    """A runner for SFT training."""
    def run(self, config: PPOConfig, compute_loss_func=None, label_smoothing=0.0):
        print(json.dumps(config.to_dict(), indent=2))
        tokenizer = get_tokenizer(
            str(config.worker.actor.model.model_path),
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        print(config.worker.actor.model.model_path, config.data.override_chat_template, config.worker.actor.model.trust_remote_code, True,)
        print(f"[DEBUG] Processor is {'not' if processor is None else ''} successfully inited.")
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping: dict[Role, type[Worker]] = {
            Role.Actor: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.Actor: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        train_dataloader, _ = create_dataloader(config.data, tokenizer, processor)
        # # 手动加载LLaVA-CC3M-Pretrain-595K数据集
        # from datasets import load_dataset
        # dataset = load_dataset("liuhaotian/LLaVA-CC3M-Pretrain-595K", split="train")
        # images_dir = "images"  # 假设images.zip已解压到此目录
        # def llava_cc3m_data_iter():
        #     for item in dataset:
        #         image_filename = item["image"]
        #         image_path = os.path.join(images_dir, image_filename)
        #         conversations = item["conversations"]
        #         yield {
        #             "image_path": image_path,
        #             "conversations": conversations,
        #             "id": item.get("id", None)
        #         }
        # train_dataloader = llava_cc3m_data_iter()
        trainer = RaySFTTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            compute_loss_func=compute_loss_func,
            label_smoothing=label_smoothing,
        )
        trainer.init_workers()
        trainer.fit()

def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())
    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)
    sft_config = OmegaConf.merge(default_config, cli_args)
    sft_config_obj = OmegaConf.to_object(sft_config)
    if isinstance(sft_config_obj, PPOConfig):
        sft_config_obj.deep_post_init()
    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "PYTHONUNBUFFERED": "1",
            }
        }
        ray.init(runtime_env=runtime_env)
    runner = Runner.remote()
    # 这里可传入自定义loss函数或label_smoothing参数
    ray.get(getattr(runner, 'run').remote(sft_config_obj, compute_loss_func=None, label_smoothing=0.1))

if __name__ == "__main__":
    main()
