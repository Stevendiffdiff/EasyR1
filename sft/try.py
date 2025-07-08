
import os
import torch
from SFTTrainerConfig import SFTTrainerConfig
from ray_SFTTrainer import SFTTrainer
from SFTData import MultiModalSFTDataset, create_dataloader
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

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

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/root/autodl-tmp/model/Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

dataloader = create_dataloader(
    data_path="sft/data.json",
    tokenizer=tokenizer,
    processor=processor,
    image_dir="/root/EasyR1/sft/image_dir",
    max_length=1024,

    train_batch_size = 2,
)
batch = next(iter(dataloader))


output_ids = model(
    **{k: v.to(model.device) for k, v in batch['model_inputs'].items()},
    # max_new_tokens=100,
    # return_dict=True
    )
print(output_ids.logits.size())
# generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(batch['model_inputs']['input_ids'], output_ids)]
# output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
# print(output_text)
###########

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "The picture bellow is labeled as A."},
#             {"type": "image", "image": "/root/EasyR1/sft/image_dir/3-1.png"},
#             {"type": "text", "text": "This picture is labeled as B."},
#             {"type": "image", "image": "/root/EasyR1/sft/image_dir/3-2.png"},
#             {"type": "text", "text": "Let a = number of people in A, b = number of people in B, what is a - b?"},
#         ],
#     }
# ]
# processed_images = [process_image_from_url("/root/EasyR1/sft/image_dir/3-1.png"), process_image_from_url("/root/EasyR1/sft/image_dir/3-2.png")]
# prompt = processor.apply_chat_template(
#                     messages, add_generation_prompt=True, tokenize=False
#                 )
# print(prompt)
# model_inputs = processor.apply_chat_template(
#     messages,
#     video_fps=1,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt"
# ).to(model.device)
# for k, v in model_inputs.items():
#     print(f"{k}, size = {v.size()}")
# output_ids = model(
#     **model_inputs
#     # return_dict=True
#     )
# print(output_ids.logits.size())
# generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_ids.logits.argmax(dim=-1))]
# output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
# print(output_text)