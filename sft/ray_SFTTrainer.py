from SFTTrainerConfig import SFTTrainerConfig
from typing import Optional
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoModelForImageToImage, PreTrainedTokenizer, ProcessorMixin, AutoModelForImageTextToText
from tqdm import tqdm

class SFTTrainer():
    def __init__(
        self,
        model_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        dataloader: StatefulDataLoader,
        config: Optional[SFTTrainerConfig] = None,
    ):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataloader = dataloader
        self.config = config
        
        self.use_peft = self.config.use_peft
        self.device = torch.cuda.current_device
        self.model = AutoModelForImageTextToText.from_pretrained(model_path).to(self.device)

    def fit(self):
        for epoch in range(self.config.max_epoch):
            data_iter = iter(self.dataloader)
            for step in tqdm(range(self.config.max_step)):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)
                