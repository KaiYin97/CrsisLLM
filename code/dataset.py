# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from encode_utils import JsonHandler, DialogueEncoder, IGNORE_IDX

class SFTDataset(Dataset):
    """
    Dataset for supervised fine-tuning (SFT) with multi-turn dialog examples.
    """
    def __init__(self, data_path: str, encoder: DialogueEncoder, max_seq_length: int):
        self.list_dialog = JsonHandler.load_json(data_path)
        self.encoder = encoder
        self.max_len = max_seq_length

    def __len__(self):
        return len(self.list_dialog)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        dialog = self.list_dialog[index]
        inputs = self.encoder.encode_dialog(dialog, self.max_len)
        return inputs

class InferenceDataset(Dataset):
    """
    A simple Dataset for inference.
    """
    def __init__(self, dialogs: List[Dict]):
        self.dialogs = dialogs

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx: int) -> Dict:
        return self.dialogs[idx]

class DataCollatorForSFT:
    """
    Collator for SFT datasets to handle padding and label alignment.
    """
    def __init__(self, encoder: DialogueEncoder):
        self.encoder = encoder
        self.pad_token_id = self.encoder.tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>") or self.encoder.tokenizer.eos_token_id

    def __call__(self, instances: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = zip(*[(instance["input_ids"], instance["labels"]) for instance in instances])

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_IDX
        )

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": input_ids_padded.ne(self.pad_token_id),
        }

def make_sft_data_module(tokenizer, args):
    encoder = DialogueEncoder(tokenizer)
    train_dataset = SFTDataset(args.train_data_path, encoder, args.max_seq_length)
    data_collator = DataCollatorForSFT(encoder)
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,  # Define if you have evaluation data
        "data_collator": data_collator
    }
