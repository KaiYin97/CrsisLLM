import io
import json
import torch
import random
from typing import Dict, List
from prompts import *

IGNORE_IDX = -100

class JsonHandler:
    @staticmethod
    def load_json(file_path: str, mode: str = "r") -> Dict:
        """
        Load a JSON file into a dictionary.
        """
        with open(file_path, mode) as f:
            return json.load(f)

class DialogueEncoder:
    """
    Handles dialogue encoding for various tasks. Encodes headers, prompts, 
    categories, user messages, and assistant responses for model input.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _get_prompt_info(self, dataname: str):
        """
        Retrieve prompt and instruction templates based on the dataset type.
        """
        if dataname == "domain":
            prompt_tasks = {1: PROMPT_TASK1, 2: PROMPT_TASK2, 3: PROMPT_TASK3, 4: PROMPT_TASK4}
            prompt_instructions = {1: PROMPT_INSTRUCTIONS_TASK1, 2: PROMPT_INSTRUCTIONS_TASK2,
                                   3: PROMPT_INSTRUCTIONS_TASK3, 4: PROMPT_INSTRUCTIONS_TASK4}
        else:  # "domain_bench"
            prompt_tasks = {1: PROMPT_TASK1, 2: PROMPT_TASK2_BENCH, 3: PROMPT_TASK3_BENCH, 4: PROMPT_TASK4_BENCH}
            prompt_instructions = {1: PROMPT_INSTRUCTIONS_TASK1, 2: PROMPT_INSTRUCTIONS_TASK2_BENCH,
                                   3: PROMPT_INSTRUCTIONS_TASK3_BENCH, 4: PROMPT_INSTRUCTIONS_TASK4_BENCH}
        return prompt_tasks, prompt_instructions

    def _encode_categories(self, task_id: int, dataname: str) -> List[int]:
        """
        Encode category tokens for tasks that require Event or Human Aid categories.
        """
        tokens = []
        if task_id in [1, 4]:
            category_type = Event_category if dataname == "domain" else Event_category_bench
            tokens += self._encode_single_category_set(category_type, is_event=True)
        if task_id in [2, 4]:
            category_type = Hum_aid_category if dataname == "domain" else Hum_aid_category_bench
            tokens += self._encode_single_category_set(category_type, is_event=False)
        return tokens

    def _encode_single_category_set(self, category_type: List, is_event: bool) -> List[int]:
        """
        Encode a category list in two forms: without descriptions and with descriptions.
        """
        tokens = []
        tokens += self.tokenizer.encode(
            self.build_category_string(category_type, with_description=False, is_event=is_event),
            add_special_tokens=False
        )
        tokens += self.tokenizer.encode("\n\n", add_special_tokens=False)
        tokens += self.tokenizer.encode(
            self.build_category_string(category_type, with_description=True, is_event=is_event),
            add_special_tokens=False
        )
        tokens += self.tokenizer.encode("\n\n", add_special_tokens=False)
        return tokens

    def build_category_string(self, category_type: List, with_description: bool = True, is_event: bool = True) -> str:
        """
        Build a formatted string for a category list, optionally with descriptions.
        """
        base_lines = [
            f"S{i+1}: {cat.name}" + (f"\n{cat.description}" if with_description else "")
            for i, cat in enumerate(category_type)
        ]
        description_only_lines = [f"S{i+1}: {cat.description}" for i, cat in enumerate(category_type)]

        if with_description:
            if is_event:
                return EVENT_CATEGORIES_DESCRP.replace("$event_categories", "\n".join(description_only_lines)).strip()
            else:
                return HUMAN_CATEGORIES_DESCRP.replace("$human_categories", "\n".join(description_only_lines)).strip()
        else:
            if is_event:
                return EVENT_CATEGORIES_DESCRP.replace("$event_categories", "\n".join(base_lines)).strip()
            else:
                return HUMAN_CATEGORIES_DESCRP.replace("$human_categories", "\n".join(base_lines)).strip()

    def build_user_message(self, user_message: Dict[str, str]) -> str:
        """
        Format a user message by inserting content into a template.
        """
        return USER_MESSAGE.replace("$message", user_message['content']).strip()

    def encode_header(self, message: Dict[str, str]) -> List[int]:
        """
        Encode message header tokens (e.g., <|start_header_id|>user<|end_header_id|>).
        """
        tokens = [self.tokenizer.convert_tokens_to_ids("<|start_header_id|>")]
        tokens += self.tokenizer.encode(message["role"], add_special_tokens=False)
        tokens.append(self.tokenizer.convert_tokens_to_ids("<|end_header_id|>"))
        tokens += self.tokenizer.encode("\n\n", add_special_tokens=False)
        return tokens

    def encode_system_prompt(self, sys_message: Dict[str, str]) -> List[int]:
        """
        Encode the system message, including special tokens like <|begin_of_text|>.
        """
        tokens = [self.tokenizer.convert_tokens_to_ids("<|begin_of_text|>")]
        tokens += self.encode_header(sys_message)
        tokens += self.tokenizer.encode(sys_message['content'].strip(), add_special_tokens=False)
        tokens.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        tokens += self.tokenizer.encode("\n", add_special_tokens=False)
        return tokens

    def encode_user_prompt(self, message: Dict[str, str], is_inference: bool, dataname: str) -> List[int]:
        """
        Encode user prompt, including task-specific instructions and optional categories.
        """
        tokens = self.encode_header(message)
        prompt_tasks, prompt_insts = self._get_prompt_info(dataname)
        task_id = message["task_id"]

        if is_inference and 4 in prompt_tasks:
            prompt_tasks[4] = PROMPT_TASK4_INFERENCE

        if task_id in prompt_tasks:
            tokens += self.tokenizer.encode(prompt_tasks[task_id].strip(), add_special_tokens=False)
        tokens += self.tokenizer.encode("\n\n", add_special_tokens=False)
        tokens += self._encode_categories(task_id, dataname)

        if is_inference or task_id == 1:
            tokens += self.tokenizer.encode(self.build_user_message(message), add_special_tokens=False)
            tokens += self.tokenizer.encode("\n\n", add_special_tokens=False)

        if task_id in prompt_insts:
            tokens += self.tokenizer.encode(prompt_insts[task_id].strip(), add_special_tokens=False)

        tokens.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        return tokens

    def encode_answer(self, answer: Dict[str, str], is_end: bool = False) -> (List[int], List[int]):
        """
        Encode assistant response, including both input and label tokens.
        """
        input_tokens = self.encode_header(answer)
        label_tokens = [IGNORE_IDX] * len(input_tokens)

        ans_toks = self.tokenizer.encode(answer['content'], add_special_tokens=False)
        input_tokens += ans_toks
        label_tokens += ans_toks

        end_id = "<|end_of_text|>" if is_end else "<|eot_id|>"
        end_token = self.tokenizer.convert_tokens_to_ids(end_id)
        input_tokens.append(end_token)
        label_tokens.append(end_token)

        return input_tokens, label_tokens

    def encode_dialog_fixed_order(self, dialog: Dict, max_len: int) -> Dict[str, torch.Tensor]:
        """
        Encode a dialogue with a fixed order: system -> user -> assistant -> ...
        """
        input_tokens = []
        label_tokens = []

        sys_tokens = self.encode_system_prompt(dialog["conversation"][0])
        input_tokens += sys_tokens
        label_tokens += [IGNORE_IDX] * len(sys_tokens)

        if dialog['type'] in ['domain', 'domain_bench']:
            for i in range(1, len(dialog["conversation"])):
                msg = dialog["conversation"][i]
                if msg['role'] == "user":
                    user_toks = self.encode_user_prompt(msg, False, dialog['type'])
                    input_tokens += user_toks
                    label_tokens += [IGNORE_IDX] * len(user_toks)
                else:
                    is_end = (i == len(dialog["conversation"]) - 1)
                    ans_input, ans_label = self.encode_answer(msg, is_end)
                    input_tokens += ans_input
                    label_tokens += ans_label

        input_tokens = input_tokens[:max_len]
        label_tokens = label_tokens[:max_len]
        return {'input_ids': torch.tensor(input_tokens), 'labels': torch.tensor(label_tokens)}

    def encode_dialog(self, dialog: Dict, max_len: int) -> Dict[str, torch.Tensor]:
        """
        Encode a dialogue where task_id=1 is completed first, followed by randomized task order.
        """
        input_tokens = []
        label_tokens = []

        sys_tokens = self.encode_system_prompt(dialog["conversation"][0])
        input_tokens += sys_tokens
        label_tokens += [IGNORE_IDX] * len(sys_tokens)

        if dialog['type'] in ['domain', 'domain_bench']:
            conversation = dialog["conversation"][1:]

            task_dict = {conversation[i]["task_id"]: conversation[i:i+2] for i in range(0, len(conversation), 2)}
            task_ids = list(task_dict.keys())

            if 1 in task_ids:
                task_ids.remove(1)
                random.shuffle(task_ids)
                task_ids.insert(0, 1)
            else:
                random.shuffle(task_ids)

            for idx, task_id in enumerate(task_ids):
                for msg in task_dict[task_id]:
                    if msg['role'] == "user":
                        user_toks = self.encode_user_prompt(msg, False, dialog['type'])
                        input_tokens += user_toks
                        label_tokens += [IGNORE_IDX] * len(user_toks)
                    else:
                        is_end = (idx == len(task_ids) - 1)
                        ans_input, ans_label = self.encode_answer(msg, is_end)
                        input_tokens += ans_input
                        label_tokens += ans_label

        input_tokens = input_tokens[:max_len]
        label_tokens = label_tokens[:max_len]
        return {'input_ids': torch.tensor(input_tokens), 'labels': torch.tensor(label_tokens)}

    def encode_inference_batch(self, dialogs: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of dialogues for inference, with padding.
        """
        input_ids_batch = []
        attention_masks_batch = []
        input_lengths = []

        for dialog in dialogs:
            tokens = self.encode_system_prompt(dialog["conversation"][0])
            if dialog['type'] == 'domain':
                tokens += self.encode_user_prompt(dialog["conversation"][1], True, dialog['type'])

            attention_mask = [1] * len(tokens)
            input_ids_batch.append(tokens)
            attention_masks_batch.append(attention_mask)
            input_lengths.append(len(tokens))

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq) for seq in input_ids_batch],
            batch_first=True,
            padding_value=self.tokenizer.eos_token_id
        )
        attention_masks_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(mask) for mask in attention_masks_batch],
            batch_first=True,
            padding_value=0
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_masks_padded,
            "input_lengths": input_lengths
        }
