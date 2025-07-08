import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import defaultdict
import re
import requests
import numpy as np
from io import BytesIO
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import fetch_video
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl.utils import torch_functional as VF


def process_image_from_url(url: str, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None):
    if url.startswith('http://') or url.startswith('https://'):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(url)
    image.load()
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = (max_pixels / (image.width * image.height)) ** 0.5
        image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))
    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = (min_pixels / (image.width * image.height)) ** 0.5
        image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def process_video_from_url(url: str, min_pixels: Optional[int] = None, max_pixels: Optional[int] = None, video_fps: float = 2.0, return_fps: bool = False):
    response = requests.get(url)
    video_bytes = BytesIO(response.content)
    vision_info = {"video": video_bytes, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)

def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    model_inputs, loss_inputs, raw_inputs = {}, {}, {}

    for key, value in tensors.items():
        if key in ['input_ids', 'attention_mask']:
            model_inputs[key] = torch.stack(value, dim=0)
        elif key in ['pixel_values', 'image_grid_thw']:
            model_inputs[key] = torch.concat(value, dim=0)
        elif key in ['selective_mask', 'labels']:
            loss_inputs[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        raw_inputs[key] = np.array(value, dtype=object)

    return dict(
        model_inputs=model_inputs,
        loss_inputs=loss_inputs,
        raw_inputs=raw_inputs,
    )

class MultiModalSFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        max_length: int = 1024,

        seq_key: str = "sequence",
        image_key: str = "image",
        video_key: str = "video",
        image_dir: Optional[str] = None,
        video_dir: Optional[str] = None,

        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        video_fps: float = 2.0,
        truncation: str = "right",
        # filter_overlong_prompts: bool = True,
        # filter_overlong_prompts_workers: int = 16,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

        self.seq_key = seq_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_dir = video_dir

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.video_fps = video_fps
        self.truncation = truncation

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.data = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.data = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.data = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        # if filter_overlong_prompts:
        #     self.dataset = self.dataset.filter(
        #         self._filter_overlong_prompts,
        #         desc="Filtering overlong prompts",
        #         num_proc=filter_overlong_prompts_workers,
        #     )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        # sequence: List[Dict], each dict has 'from' and 'value'
        sequence_list = example[self.seq_key]
        images = example.get(self.image_key, [])
        # print(f"[DEBUG] images = {images}")
        videos = example.get(self.video_key, [])
        if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
            images = [os.path.join(self.image_dir, image) for image in images]
            # print(f"[DEBUG] images = {images}")
        if self.video_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
            videos = [os.path.join(self.video_dir, video) for video in videos]

        # 拼接所有value，记录每段的起止和from
        concat_text = ""
        seg_ranges = []  # (start, end, from)
        cur = 0
        for seg in sequence_list:
            value = seg["value"]
            seg_from = seg["from"]
            concat_text += value
            seg_ranges.append((cur, cur + len(value), seg_from))
            cur += len(value)

        # 解析 <image1>/<video1> 占位符
        image_pattern = re.compile(r"<image(\d+)>")
        # print(f"[DEBUG] image_pattern = {image_pattern}")
        video_pattern = re.compile(r"<video(\d+)>")
        image_indices = [int(m.group(1)) - 1 for m in image_pattern.finditer(concat_text)]
        # print(f"[DEBUG] image_indices = {image_indices}, in {concat_text}")
        video_indices = [int(m.group(1)) - 1 for m in video_pattern.finditer(concat_text)]
        sequence_for_tokenize = image_pattern.sub("<image>", concat_text)
        sequence_for_tokenize = video_pattern.sub("<video>", sequence_for_tokenize)

        # 构建 content_list
        content_list = []
        last = 0
        for m in re.finditer(r"<image>|<video>", sequence_for_tokenize):
            if m.start() > last:
                content_list.append({"type": "text", "text": sequence_for_tokenize[last:m.start()]})
            if m.group() == "<image>":
                content_list.append({"type": "image"})
            else:
                content_list.append({"type": "video"})
            last = m.end()
        if last < len(sequence_for_tokenize):
            content_list.append({"type": "text", "text": sequence_for_tokenize[last:]})
        messages = [{"role": "user", "content": content_list}]

        # 下载并处理图片/视频
        processed_images = [process_image_from_url(images[i], self.min_pixels, self.max_pixels) for i in image_indices] if images else None
        # print(f"[DEBUG] processed_images = {processed_images}")
        processed_videos = [process_video_from_url(videos[i], self.min_pixels, self.max_pixels, self.video_fps) for i in video_indices] if videos else None

        # 编码
        model_inputs = None
        if self.processor is not None and callable(self.processor):
            if processed_images and not processed_videos:
                prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                    processor_kwargs={}, mm_load_kwargs={}, template_kwargs={}
                )
                model_inputs = self.processor(
                    processed_images, [prompt], add_special_tokens=False, return_tensors="pt",
                    processor_kwargs={}, mm_load_kwargs={}
                )
                # print(f"[DEBUG] templated prompt = {prompt}")
            elif processed_videos and not processed_images:
                prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                    processor_kwargs={}, mm_load_kwargs={}, template_kwargs={}
                )
                model_inputs = self.processor(
                    videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt",
                    processor_kwargs={}, mm_load_kwargs={}
                )
            elif processed_images and processed_videos:
                prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                    processor_kwargs={}, mm_load_kwargs={}, template_kwargs={}
                )
                model_inputs = self.processor(
                    images=processed_images, videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt",
                    processor_kwargs={}, mm_load_kwargs={}
                )
        if model_inputs is None:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")

        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]

        # pad/truncation
        truncation_mode = self.truncation if self.truncation in ("left", "right", "error") else "right"
        pad_token_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else 0
        input_ids, attention_mask, _ = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            max_length=self.max_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation=truncation_mode,
        )

        # selective_mask: 只对gpt生成的token为1
        selective_mask = torch.zeros_like(input_ids)
        try:
            encoding = self.tokenizer(sequence_for_tokenize, add_special_tokens=False, return_offsets_mapping=True)
            offsets = list(encoding["offset_mapping"])
            for i, (start, end) in enumerate(offsets):
                for seg_start, seg_end, seg_from in seg_ranges:
                    if start >= seg_start and end <= seg_end and seg_from == "gpt":
                        selective_mask[i+1] = 1  # +1是因为通常有BOS token
                        break
        except Exception:
            cur = 0
            for seg_start, seg_end, seg_from in seg_ranges:
                if seg_from == "gpt":
                    for i in range(seg_start, seg_end):
                        selective_mask[i] = 1

        # labels
        labels = input_ids.clone()
        labels[selective_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_grid_thw": model_inputs.get("image_grid_thw", None),
            "video_grid_thw": model_inputs.get("video_grid_thw", None),
            "pixel_values": model_inputs.get("pixel_values", None),
            "labels": labels,
            "selective_mask": selective_mask,
            "raw_sequence": sequence_list,
            "images": processed_images,
            "videos": processed_videos,
        }

def create_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin] = None,
    max_length: int = 1024,

    seq_key: str = "sequence",
    image_key: str = "image",
    video_key: str = "video",
    image_dir: Optional[str] = None,
    video_dir: Optional[str] = None,

    format_prompt: Optional[str] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    video_fps: float = 2.0,
    truncation: str = "right",

    # dataloader params
    shuffle: bool = True,
    seed: int = 0,
    train_batch_size: int = 64,
):
    dataset = MultiModalSFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        processor=processor,
        max_length=max_length,

        seq_key=seq_key,
        image_key=image_key,
        video_key=video_key,
        image_dir=image_dir,
        video_dir=video_dir,

        format_prompt=format_prompt,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        video_fps=video_fps,
        truncation=truncation,
    )
    # use sampler for better ckpt resume
    if shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    train_dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    assert len(train_dataloader) >= 1, "Invalid train_dataloader! The size of the loader = 0!"
    print(f"Size of train dataloader: {len(train_dataloader)}")
    return train_dataloader
