#!/bin/bash


set +e

deepspeed -- ./code/train.py --train_args_file './code/train_args.json' --lora_r 128 --lora_alpha 128 --lora_dropout 0.5 --loss_name "lora_128_128_05" --train_mode "lora" 





