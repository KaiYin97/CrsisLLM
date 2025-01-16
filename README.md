# CrsisLLM
source code for fine-tuing LLama 3.1 for multi-task social media txt processing for Disaster Management. 
This work is conducted based on our previous work: Crisissense-llm: Instruction fine-tuned large language model for multi-label social media text classification in disaster informatics.
In this work, we do multi-label text classification and NER for location identification. 
# Traning and inference data 
All training data is in /data/dialog.json, while inference data is in dialog4inference_train_part.json.
# Training 
The training code is in /code/train.py which uses Deepspeed-zero-stage-3. I considered full parameter tuning and LORA fine tuning. 
Please refer to /code/configs for configuration of training settings. 
# Inference 
For inference, please refer to /code/inference.py
