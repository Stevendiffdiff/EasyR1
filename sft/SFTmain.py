import os
import torch
from ray_SFTTrainer import SFTTrainer
from SFTData import MultiModalSFTDataset
from transformers import AutoProcessor, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-tmp/model/Qwen/Qwen2.5-VL-3B-Instruct",
    trust_remote_code=False,
    use_fast=True,
)
processor = AutoProcessor.from_pretrained(
    "/root/autodl-tmp/model/Qwen/Qwen2.5-VL-3B-Instruct",
    trust_remote_code=False,
    use_fast=True,
)

dataset = MultiModalSFTDataset(
    data_path="sft/data.json",
    tokenizer=tokenizer,
    processor=processor,
    image_dir="/root/EasyR1/sft/image_dir",
    max_length=2048,
)
print(dataset.__len__())
item = dataset.__getitem__(1)
print(item['selective_mask'].sum())
print((item['labels'] != -100).sum())
for k, v in item.items():
    print(k, ': ', v)
    if torch.is_tensor(v):
        print(v.size())





dataloader = None
data_collater = None
trainer = SFTTrainer(

)
trainer.fit()