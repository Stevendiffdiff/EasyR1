from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple
from peft import LoraConfig

@dataclass
class SFTTrainerConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 1e-6
    beta_1: float = 0.99
    beta_2: float = 0.95
    max_step: int = 100
    max_epoch: int = 5

    use_peft: bool = True
    peft_config: Optional[LoraConfig] = None

    