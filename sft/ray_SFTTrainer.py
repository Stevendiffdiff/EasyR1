from SFTTrainerConfig import SFTTrainerConfig
from typing import Optional, Callable
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.optim import Optimizer, AdamW
from transformers import AutoModelForImageTextToText, PreTrainedTokenizer, ProcessorMixin
from transformers.trainer_pt_utils import LabelSmoother
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

class SFTTrainer():
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        dataloader: StatefulDataLoader,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[str] = None,
        loss_fn: Optional[Callable] = None,
        config: Optional[SFTTrainerConfig] = None,
        optimizer: Optional[Optimizer] = None
    ):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataloader = dataloader
        self.config = config if config is not None else SFTTrainerConfig()
        self.loss_fn = LabelSmoother(
            self.config.label_smoother_epsilon, 
            self.config.label_smoother_ignore_index
        ) if loss_fn is None else loss_fn
        
        self.device = torch.cuda.current_device()
        self.use_peft = self.config.use_peft
        if model is None:
            self.model = AutoModelForImageTextToText.from_pretrained(model_path).to(self.device)
            if self.use_peft:
                lora_config = self.config.peft_config
                assert lora_config is not None, "Fail to construct PEFT Model, because PEFT config has not been provided, while use_peft has been set as 'True'!"
                self.model = get_peft_model(self.model, lora_config)
        else:
            self.model = model

        self.optimizer = optimizer if optimizer is not None else AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            betas=(self.config.beta_1, self.config.beta_2)
        )

    def prepare_data(self, batch: dict):
        for key in batch.keys():
            for k, v in batch[key].items():
                if torch.is_tensor(v):
                    batch[key][k] = v.to(self.device)

        return batch

    def fit(self):
        for epoch in range(self.config.max_epoch):
            data_iter = iter(self.dataloader)
            for step in tqdm(range(self.config.max_step), desc=f"Epoch {epoch + 1} / {self.config.max_epoch}"):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch: dict = next(data_iter)
                
                batch = self.prepare_data(batch)
                model_inputs = batch.pop('model_inputs')
                loss_inputs = batch.pop('loss_inputs')
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                )
                loss = self.loss_fn(outputs, loss_inputs['labels'], shift_labels=True)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(f"[DEBUG] {epoch + 1}-{step + 1}, loss = {loss.cpu().item()}")
    
    def save_model(self, path: str):
        self.model.save_pretrained(path)
        print(f"The trained model has been saved to {path}. The type of the model is {self.model.__class__}.")
