## CrsisLLM
source code for fine-tuing LLama 3.1 for multi-task social media txt processing for Disaster Management. Instruction fine-tuned large language model for multi-label social media text classification in disaster informatics

## Introduction 
This work is conducted based on our previous work [CrisisSense-LLM: Instruction Fine-Tuned Large Language Model for Multi-label Social Media Text Classification in Disaster Informatics](https://arxiv.org/abs/2406.15477). For this work, I fine-tuned LLama 3.1-8B for multi-label text classification and NER for location identification of open-accessible disaster-related Twitter dataset. The fine-tuning is conducted through multi-turn Supervised-Fine-Tuning with each turn dealing with one specific task (event category classification, or humanitarian aid category classification, or location entities recognization). 
## Prompt design
Please refer to the official document of special tokens of LLama 3.1 in https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/. This prompt design gains some experience from [Llama Guard](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/) prompts design which is officially releaed in: https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_recipes/datasets/toxicchat_dataset.py. 
## Prompt Example

Here is a detailed example of the fine-tuning prompt:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Task: For the below User message, check which event category it belongs to with the below EVENT CATEGORIES given the DESCRIPTION of each EVENT CATEGORY.

<BEGIN EVENT CATEGORIES DESCRIPTION>
S1: Wildfires: Uncontrolled fires that spread rapidly through vegetation, often caused by natural or human factors.
S2: Cyclone: a large-scale air mass that rotates around a strong center of low atmospheric pressure, typically leading to strong winds and heavy rain.
S3: Earthquake: sudden shaking of the ground caused by the movement of tectonic plates beneath the Earth's surface
S4: Hurricane: a powerful tropical storm characterized by strong winds, heavy rain, and potential for severe coastal damage.
S5: Floods: the overflow of water onto normally dry land, often caused by excessive rainfall, river overflow, or storm surges.
<END EVENT CATEGORIES DESCRIPTION>

<BEGIN USER MESSAGE>
A hurricane happened in Houston
<END USER MESSAGE>

You should provide your response in JSON format, with the following structure: {"EVENT_CATEGORY": event category the User Message belongs to}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{"event_type": "hurricane"}<|eot_id|><|start_header_id|>user<|end_header_id|>

Task: For the above User message, find all location entities contained in it.

You should provide your response in JSON format with all location entities in a List object, with the following structure: {"LOCATIONS": List of location entities the User Message contained}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{"LOCATIONS": "yyy"}<|eot_id|><|start_header_id|>user<|end_header_id|>

Task: For the above User message, first check which event category it belongs to with the below EVENT CATEGORIES given the DESCRIPTION of each EVENT CATEGORY;
second, check which humanitarian aid category it belongs to with the below HUMANITARIAN AID CATEGORIES given the DESCRIPTION of each HUMANITARIAN AID CATEGORY;
finally find all location entities contained in it.

<BEGIN HUMANITARIAN AID CATEGORIES DESCRIPTION>
S1: caution_and_advice: Guidance or warnings provided to the public to help them stay safe or prepare for a disaster.
S2: displaced_people_and_evacuations: Information about people who have been forced to leave their homes due to a disaster and efforts to relocate them.
S3: infrastructure_and_utility_damage: Reports on the destruction or impairment of buildings, roads, power lines, and other essential services due to a disaster.
S4: injured_or_dead_people: Information regarding casualties, including the number of people injured or killed due to a disaster.
S5: missing_or_found_people: Reports on individuals who are unaccounted for or have been located after being reported missing during a disaster.
S6: requests_or_urgent_needs: Appeals for immediate assistance, resources, or support to address critical situations during or after a disaster.
S7: rescue_volunteering_or_donation_effort: Information about rescue operations, volunteer involvement, and efforts to collect donations to support disaster-affected areas.
<END HUMANITARIAN AID CATEGORIES DESCRIPTION>

You should provide your response in JSON format, with the following structure:
{"EVENT_CATEGORY": event category the User Message belongs to, "HUMANITARIAN_AID": humanitarian aid category the User Message belongs to, "LOCATIONS": List of location entities the User Message contained}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{"event_type": "hurricane", "human_aid": "yyy", "LOCATIONS": "yyy"}<|eot_id|><|start_header_id|>user<|end_header_id|>

Task: For the above User message, check which humanitarian aid category it belongs to with the below HUMANITARIAN AID CATEGORIES given the DESCRIPTION of each HUMANITARIAN AID CATEGORY.

You should provide your response in JSON format, with the following structure: {"HUMANITARIAN_AID": humanitarian aid category the User Message belongs to}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{"human_aid": "yyy"}<|eot_id|>
```


## Traning and inference data 
The training and inference data is originated from IDRISI https://github.com/rsuwaileh/IDRISI which contains human labeled location, human aid information. I did some post-processing to make it more suitable for the current work. 
All training data is in /data/dialog.json, while inference data is in dialog4inference_train_part.json.
## Training setting 
The training code is in /code/train.py. This project uses Deepspeed-zero-stage-3 in A-100-40Gb GPUs with mixed precision training to accelerate training. Both full parameter tuning and LORA fine tuning are considered. Different hyperparameter combinations of LORA fine-tuning are considered. You could also consider other PEFTs (such as QLORA, etc.) based on my current training code. Please refer to /code/configs for configuration details of the training hyperparameter. 
## Inference 
For inference, please refer to /code/inference.py
## Instruction
You can use run_deepspeed.sh for training.

## Citation

If you find this repository helpful, please kindly consider citing the corresponding paper as shown below. Thanks!

```bibtex
@article{yin2024crisissense,
    title={Crisissense-LLM: Instruction Fine-tuned Large Language Model for Multi-label Social Media Text Classification in Disaster Informatics},
    author={Yin, Kai and Liu, Chengkai and Mostafavi, Ali and Hu, Xia},
    journal={arXiv preprint arXiv:2406.15477},
    year={2024}
}



