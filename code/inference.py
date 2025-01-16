# inference.py

import os
import json
import torch
import argparse
from typing import List, Dict
from tqdm import tqdm
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

from encode_utils import JsonHandler, DialogueEncoder, InferenceDataset


class InferenceManager:
    """
    InferenceManager handles the loading of models (with optional LoRA adapters),
    processing of input dialogues, performing batch inference, and saving the results.
    """

    def __init__(self, args):
        self.directory = args.directory
        self.lora_folders = args.lora_folders
        self.batch_size = args.batch_size
        self.output_base_dir = "./output/PEFT_infer_res"
        self.model = None
        self.tokenizer = None
        self.encoder = None

    def load_model(self, is_lora: bool, lora_adapter_path: str = ""):
        """
        Load the base model and optionally apply a LoRA adapter.
        """
        base_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            cache_dir='./model/',
            torch_dtype=torch.float16,
            use_auth_token=True  
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            cache_dir='./model/',
            use_fast=True
        )

        if is_lora and lora_adapter_path:
            peft_config = PeftConfig.from_pretrained(lora_adapter_path)
            self.model = PeftModel.from_pretrained(
                self.model, 
                lora_adapter_path, 
                device_map="auto"
            )
        self.encoder = DialogueEncoder(self.tokenizer)

    def find_lora_checkpoints(self, directory: str, lora_folders_to_search: List[str]) -> List[Dict[str, str]]:
        """
        Find all LoRA checkpoint folders within specified directories.
        """
        result = []
        for folder in lora_folders_to_search:
            lora_folder_path = os.path.join(directory, folder)
            if os.path.isdir(lora_folder_path):
                checkpoint_folders = [
                    subfolder for subfolder in os.listdir(lora_folder_path)
                    if subfolder.startswith("checkpoint") and 
                       os.path.isdir(os.path.join(lora_folder_path, subfolder))
                ]
                for checkpoint_folder in checkpoint_folders:
                    full_path = os.path.join(lora_folder_path, checkpoint_folder)
                    checkpoint_number = checkpoint_folder.split("-")[-1] if "-" in checkpoint_folder else ""
                    save_path = os.path.join(
                        self.output_base_dir, 
                        folder, 
                        f"ckpt_{checkpoint_number}.json"
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    result.append({"path": full_path, "save_path": save_path})
            else:
                print(f"Folder '{folder}' does not exist or is not a directory.")
        return result

    def batch_inference(self, dialogs: List[Dict]) -> List[str]:
        """
        Perform batch inference on a list of dialogs.
        """
        dataset = InferenceDataset(dialogs)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=lambda x: x
        )

        all_predictions = []

        for batch_dialogs in tqdm(dataloader, desc="Processing batches"):
            input_data = self.encoder.encode_inference_batch(batch_dialogs)
            input_ids = input_data["input_ids"].to("cuda")
            attention_mask = input_data["attention_mask"].to("cuda")
            input_lengths = input_data["input_lengths"]

            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                pad_token_id=self.tokenizer.eos_token_id  # Ensure pad_token_id is set
            )

            output_trimmed = [
                output[i, input_lengths[i]:] for i in range(len(output))
            ]

            pred_texts = self.tokenizer.batch_decode(
                output_trimmed, 
                skip_special_tokens=True
            )
            all_predictions.extend(pred_texts)

        return all_predictions

    def perform_inference(self, lora_path: str, output_json_path: str):
        """
        Perform inference for a single LoRA checkpoint and save the results.
        """
        self.load_model(is_lora=True, lora_adapter_path=lora_path)

        dialogs = JsonHandler.load_json('./data/dialog4inference_train_part.json')

        all_generated_texts = self.batch_inference(dialogs)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_generated_texts, f, ensure_ascii=False, indent=4)

        print(f"Generated results have been saved to {output_json_path}")

    def run_inference(self):
        """
        Orchestrate the inference process across multiple LoRA checkpoints.
        """
        checkpoints_info = self.find_lora_checkpoints(self.directory, self.lora_folders)
        for info in checkpoints_info:
            self.perform_inference(info['path'], info['save_path'])


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Find and process LoRA checkpoints for inference.")
    parser.add_argument(
        "--directory", 
        type=str, 
        default="./output/", 
        help="Directory to search for LoRA folders."
    )
    parser.add_argument(
        "--lora_folders", 
        nargs='+', 
        required=True, 
        help="List of LoRA folders to search."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=20, 
        help="Batch size for inference."
    )
    return parser.parse_args()


def main():
    """
    Main function to initiate the inference process.
    """
    args = parse_arguments()
    inference_manager = InferenceManager(args)
    inference_manager.run_inference()


if __name__ == "__main__":
    main()
