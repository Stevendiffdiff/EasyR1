import os
import torch
from SFTTrainerConfig import SFTTrainerConfig
from ray_SFTTrainer import SFTTrainer
from SFTData import MultiModalSFTDataset, create_dataloader
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText
from peft import LoraConfig, PeftModel

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

dataloader = create_dataloader(
    data_path="sft/data.json",
    tokenizer=tokenizer,
    processor=processor,
    image_dir="/root/EasyR1/sft/image_dir",
    max_length=1024,

    train_batch_size = 2,
)

config = SFTTrainerConfig(
    use_peft=True,
    peft_config=LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.05,
        # base_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        # bias="none",
        # task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    ),
    learning_rate=2e-4,
    weight_decay=0.01,
    beta_1=0.9,
    beta_2=0.999,
    max_step=4,
    max_epoch=1,
    label_smoother_epsilon=0.1,
    label_smoother_ignore_index=-100,
)

trainer = SFTTrainer(
    tokenizer,
    processor, 
    dataloader,
    model_path="/root/autodl-tmp/model/Qwen/Qwen2.5-VL-3B-Instruct",
    config=config,
)
trainer.fit()
trainer.save_model("/root/autodl-tmp/checkpoints/sft/save_model_1")

base_model = AutoModelForImageTextToText.from_pretrained("/root/autodl-tmp/model/Qwen/Qwen2.5-VL-3B-Instruct")
retrieved_model = PeftModel.from_pretrained(base_model, "/root/autodl-tmp/checkpoints/sft/save_model_1")
print(f"Successfully retrieve the trained model! The model summary is \n {retrieved_model}.")