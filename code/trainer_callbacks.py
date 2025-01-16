# trainer_callbacks.py

import os
from transformers import TrainerCallback

class SaveLossCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.losses = []
        self.loss_file = os.path.join(self.output_dir, "losses.txt")
        with open(self.loss_file, "w") as f:
            f.write("")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])
            with open(self.loss_file, "a") as f:
                f.write(f"Step {state.global_step}: {logs['loss']}\n")

class SaveLRCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.lrs = []
        self.lr_file = os.path.join(self.output_dir, "learning_rates.txt")
        # Clear the file if it exists
        with open(self.lr_file, "w") as f:
            f.write("")

    def on_step_end(self, args, state, control, **kwargs):
        optimizer = kwargs.get("optimizer", None)
        if optimizer:
            lr = optimizer.param_groups[0]['lr']
            self.lrs.append((state.global_step, lr))
            with open(self.lr_file, "a") as f:
                f.write(f"Step {state.global_step}: {lr}\n")

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int, min_delta: float, output_dir: str):
        self.patience = patience
        self.min_delta = min_delta
        self.output_dir = output_dir
        self.best_loss = None
        self.counter = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            current_loss = logs["loss"]
            if self.best_loss is None or (self.best_loss - current_loss) > self.min_delta:
                self.best_loss = current_loss
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= self.patience:
                print(f"Early stopping triggered. No improvement in loss for {self.patience} steps.")
                control.should_training_stop = True
