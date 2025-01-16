import os
import json
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments as HFTrainingArguments

@dataclass
class DataArguments:
    """
    Custom arguments for data loading.
    """
    data_path: str = field(
        default="./data/dialog_combined.json", 
        metadata={"help": "Path to the training data."}
    )

@dataclass
class TrainingArguments(HFTrainingArguments):
    """
    Extends HF TrainingArguments with custom fields.
    """
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    max_seq_length: int = field(default=128000)
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    train_mode: str = field(default="full")
    train_data_path: str = field(default="data/dialog.json")
    lora_alpha: int = field(default=128)
    lora_r: int = field(default=64)
    lora_dropout: float = field(default=0.5)
    loss_name: str = field(default="full_tune.text")
