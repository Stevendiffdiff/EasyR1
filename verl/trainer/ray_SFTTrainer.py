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
"""
SFT Trainer with Ray-based single controller. Supports multi-modal input (dict), output is always text.
"""

import os
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Type, Callable

import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
from torch.utils.tensorboard import SummaryWriter
from transformers.trainer_pt_utils import LabelSmoother

from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils.logger import Tracker
from .config import PPOConfig
from ..workers.fsdp_workers import FSDPWorker


class Role(IntEnum):
    Actor = auto()

@dataclass
class ResourcePoolManager:
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])


class RaySFTTrainer:
    """
    SFT Trainer for multi-modal input (dict), output is always text.
    """
    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: Optional[StatefulDataLoader],
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        compute_loss_func: Optional[Callable] = None,
        label_smoothing: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader  # 可为None，当前SFTTrainer不使用
        self.config = config
        self.compute_loss_func = compute_loss_func
        self.label_smoother = LabelSmoother(epsilon=label_smoothing) if label_smoothing > 0 else None
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.tb_writer = None

    def init_workers(self) -> None:
        print("[DEBUG] Entering RaySFTTrainer.init_workers")
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor], config=self.config.worker, role="actor"
        )
        print(f"[DEBUG] actor_cls: {actor_cls}")
        self.resource_pool_to_cls[resource_pool]["actor"] = actor_cls
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            print(f"[DEBUG] Creating colocated worker class for pool {resource_pool}")
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            print(f"[DEBUG] worker_dict_cls: {worker_dict_cls}")
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            print(f"[DEBUG] wg_dict: {wg_dict}")
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            print(f"[DEBUG] spawn_wg: {spawn_wg}")
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)
        self.actor_wg = all_wg["actor"]
        print(f"[DEBUG] self.actor_wg: {self.actor_wg}")
        print(f"[DEBUG] self.actor_wg._workers: {getattr(self.actor_wg, '_workers', None)}")
        print(f"[DEBUG] dir(self.actor_wg._workers[0]): {dir(self.actor_wg._workers[0]) if hasattr(self.actor_wg, '_workers') and self.actor_wg._workers else 'N/A'}")
        self.actor_wg.init_model()

    def _save_checkpoint(self, global_step: int) -> None:
        save_checkpoint_path = self.config.trainer.save_checkpoint_path or "checkpoints/default"
        folder_path = os.path.join(str(save_checkpoint_path), f"global_step_{global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)
        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

    def _load_checkpoint(self) -> Optional[int]:
        if self.config.trainer.load_checkpoint_path is None:
            return None
        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_wg.load_checkpoint(actor_path)
        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")
        return global_step

    def compute_loss(self, outputs, labels):
        if self.compute_loss_func is not None:
            return self.compute_loss_func(outputs, labels)
        elif self.label_smoother is not None:
            return self.label_smoother(outputs, labels, shift_labels=True)
        else:
            # default: standard cross-entropy
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            # logits: (B, L, V), labels: (B, L)
            return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    def fit(self):
        print("[DEBUG] Entering RaySFTTrainer.fit")
        tb_log_dir = os.path.join("logs", "tensorboard", str(getattr(self.config.trainer, "experiment_name", "default")))
        self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
        # TrainerConfig.logger: Tuple[str], 需转为list[str]
        loggers = list(self.config.trainer.logger) if isinstance(self.config.trainer.logger, tuple) else self.config.trainer.logger
        self.logger = Tracker(loggers=loggers, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.config.trainer.max_steps or len(self.train_dataloader) * self.config.trainer.total_epochs), desc="Running step", position=0)
        self._load_checkpoint()
        self.data_iterator = iter(self.train_dataloader)
        model = self.actor_wg  # Ray worker group
        print(f"[DEBUG] In fit, self.actor_wg: {self.actor_wg}")
        print(f"[DEBUG] In fit, self.actor_wg._workers: {getattr(self.actor_wg, '_workers', None)}")
        print(f"[DEBUG] In fit, dir(self.actor_wg._workers[0]): {dir(self.actor_wg._workers[0]) if hasattr(self.actor_wg, '_workers') and self.actor_wg._workers else 'N/A'}")
        for epoch in range(self.config.trainer.total_epochs):
            for batch in self.train_dataloader:
                self.global_step += 1
                # batch: dict, keys可能有input_ids, attention_mask, images, videos, labels等
                # 只需保证模型forward能处理dict输入
                model_inputs = batch.copy()
                print(f"[DEBUG] Calling forward_and_loss on actor_wg at step {self.global_step}")
                labels = model_inputs.pop("ground_truth")
                try:
                    outputs = self.actor_wg.forward_and_loss(model_inputs, labels=labels, return_outputs=True)
                except Exception as e:
                    print(f"[ERROR] Exception when calling forward_and_loss: {e}")
                    print(f"[DEBUG] type(self.actor_wg._workers[0]): {type(self.actor_wg._workers[0])}")
                    print(f"[DEBUG] dir(self.actor_wg._workers[0]): {dir(self.actor_wg._workers[0])}")
                    raise
                # print(f"[DEBUG] Labels are {labels}, type = {type(labels)}, {type(labels[0])}")
                # print(f"[DEBUG] Outputs are {outputs}")
                # seq_len = outputs[0]['logits'].shape[1]
                # print(f"[DEBUG] Seq-length = {seq_len}")
                # labels_tokenized = self.tokenizer(
                #     labels.tolist(),  # 如果labels是list[str]，否则labels.tolist()
                #     padding='max_length',
                #     truncation=True,
                #     max_length=seq_len,
                #     return_tensors="pt"
                # )["input_ids"]
                # loss = self.compute_loss(outputs, labels_tokenized)
                loss = outputs[0]['loss']
                print(f"[DEBUG] loss is {loss}")
                if loss is not None:
                    loss.backward()
                    for para in model[0].parameters():
                        print(para.grad_norm())
                    model.optimizer.step()
                # 日志
                self.tb_writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.logger.log({"train/loss": loss.item()}, step=self.global_step)
                main_tqdm.update()
                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    self._save_checkpoint(self.global_step)
            # 可选：每个epoch保存一次
            if self.config.trainer.save_freq <= 0:
                self._save_checkpoint(self.global_step)
        self.tb_writer.close()
