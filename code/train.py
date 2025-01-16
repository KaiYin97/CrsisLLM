# train.py

import os
import json
import argparse
import torch
from transformers import (
    set_seed,
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer
)
from peft import LoraConfig, get_peft_model

from config import TrainingArguments
from dataset import make_sft_data_module
from trainer_callbacks import SaveLossCallback, SaveLRCallback, EarlyStoppingCallback


class ModelTrainer:
    """
    Manages the entire training workflow including argument parsing, model setup,
    data preparation, callback configuration, and training execution.
    """

    def __init__(self):
        self.args = self.setup_arguments()
        self.training_args = self.load_training_args()
        self.tokenizer = None
        self.model = None
        self.data_module = None
        self.callbacks = []

    def setup_arguments(self):
        """
        Parse command-line arguments for training configurations.
        """
        parser = argparse.ArgumentParser(description="Training script for LLM fine-tuning.")
        parser.add_argument("--train_args_file", type=str, default='./code/configs/train_args_lora.json',
                            help="Path to training arguments JSON file.")
        parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank.")
        parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha.")
        parser.add_argument("--lora_dropout", type=float, default=0.5, help="LoRA dropout.")
        parser.add_argument("--loss_name", type=str, default="full_tune", help="Name for the loss tracking.")
        parser.add_argument("--train_mode", type=str, default="lora", help="Training mode: 'lora' or 'full'.")
        parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
        return parser.parse_args()

    def load_training_args(self):
        """
        Load training configurations from a JSON file and override them with command-line arguments.
        """
        hf_parser = HfArgumentParser(TrainingArguments)
        training_args, = hf_parser.parse_json_file(json_file=self.args.train_args_file)

        training_args.lora_r = self.args.lora_r
        training_args.lora_alpha = self.args.lora_alpha
        training_args.lora_dropout = self.args.lora_dropout
        training_args.loss_name = self.args.loss_name
        training_args.train_mode = self.args.train_mode

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(self.args.train_args_file, "r") as f:
            train_args_json = json.load(f)
        with open(os.path.join(training_args.output_dir, 'train_args.json'), "w") as f:
            json.dump(train_args_json, f, indent=4)

        set_seed(training_args.seed)

        return training_args

    def load_model(self):
        """
        Load the base model and tokenizer. Optionally configure LoRA for fine-tuning.
        """
        if self.training_args.train_mode == "lora":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.training_args.model_name_or_path,
                cache_dir=self.training_args.cache_dir
            )
            
            peft_config = LoraConfig(
                r=self.training_args.lora_r,
                lora_alpha=self.training_args.lora_alpha,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                'gate_proj', 'down_proj', 'up_proj', 'lm_head'],
                lora_dropout=self.training_args.lora_dropout
            )
            self.model = get_peft_model(self.model, peft_config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3.1-8B",
                cache_dir=self.training_args.cache_dir
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.training_args.model_name_or_path if self.training_args.train_mode == "lora" else "meta-llama/Meta-Llama-3.1-8B",
            cache_dir=self.training_args.cache_dir,
            use_fast=True
        )

    def prepare_data(self):
        """
        Prepare the training dataset and data collator.
        """
        self.data_module = make_sft_data_module(self.tokenizer, self.training_args)

    def setup_callbacks(self):
        """
        Initialize callbacks for saving loss, learning rate, and early stopping.
        """
        self.callbacks.extend([
            SaveLossCallback(self.training_args.output_dir),
            SaveLRCallback(self.training_args.output_dir),
            EarlyStoppingCallback(
                patience=20,
                min_delta=0.001,
                output_dir=self.training_args.output_dir
            )
        ])

    def calculate_save_steps(self):
        """
        Dynamically calculate save steps based on dataset size and training configurations.
        """
        data_num = len(self.data_module['train_dataset']) * self.training_args.num_train_epochs
        total_gpus = max(torch.cuda.device_count(), 1)
        total_steps = data_num // (
            self.training_args.per_device_train_batch_size *
            self.training_args.gradient_accumulation_steps *
            total_gpus
        )
        self.training_args.warmup_ratio = self.training_args.num_train_epochs * 1e-2

        self.training_args.save_steps = int(total_steps * (0.02 if self.training_args.train_mode == "lora" else 0.01))

    def train_model(self):
        """
        Train the model using the Trainer API and save the final model.
        """
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            **self.data_module,
            callbacks=self.callbacks
        )

        trainer.train()
        trainer.save_state()
        trainer.save_model(output_dir=self.training_args.output_dir)

        del self.model
        del trainer
        torch.cuda.empty_cache()

    def run(self):
        """
        Execute the full training process: load model, prepare data, set up callbacks,
        calculate save steps, and train the model.
        """
        self.load_model()
        self.prepare_data()
        self.setup_callbacks()
        self.calculate_save_steps()
        self.train_model()


def main():
    """
    Entry point for the training script.
    """
    trainer = ModelTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
