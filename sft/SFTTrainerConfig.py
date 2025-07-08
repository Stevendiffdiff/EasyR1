from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple
from peft import LoraConfig

@dataclass
class SFTTrainerConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 1e-6
    beta_1: float = 0.99
    beta_2: float = 0.95
    max_step: int = 4
    max_epoch: int = 5

    use_peft: bool = False
    peft_config: Optional[LoraConfig] = None

    label_smoother_epsilon: float = 0.1
    label_smoother_ignore_index: int = -100

    def post_init(
        self,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        beta_1: Optional[float] = None,
        beta_2: Optional[float] = None,
        max_step: Optional[int] = None,
        max_epoch: Optional[int] = None,
        use_peft: Optional[bool] = None,
        peft_config: Optional['LoraConfig'] = None,
        label_smoother_epsilon: Optional[float] = None,
        label_smoother_ignore_index: Optional[int] = None,
    ):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if weight_decay is not None:
            self.weight_decay = weight_decay
        if beta_1 is not None:
            self.beta_1 = beta_1
        if beta_2 is not None:
            self.beta_2 = beta_2
        if max_step is not None:
            self.max_step = max_step
        if max_epoch is not None:
            self.max_epoch = max_epoch
        if use_peft is not None:
            self.use_peft = use_peft
        if peft_config is not None:
            self.peft_config = peft_config
        if label_smoother_epsilon is not None:
            self.label_smoother_epsilon = label_smoother_epsilon
        if label_smoother_ignore_index is not None:
            self.label_smoother_ignore_index = label_smoother_ignore_index
