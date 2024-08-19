import lightning as L
import torch
from torch.utils.data import DataLoader
import re
from nltk import edit_distance
import numpy as np
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from huggingface_hub import HfApi

from .settings import *

api = HfApi()

def get_callback():
    early_stop_callback = EarlyStopping(
        monitor="val_edit_distance", patience=3, verbose=False, mode="min"
    )
    return early_stop_callback


def get_LlavaModelPLModule(train_dataset, val_dataset, train_collate_fn, eval_collate_fn, processor, model):
    
    class LlavaModelPLModule(L.LightningModule):
        def __init__(self, config, processor, model):
            super().__init__()
            self.config = config
            self.processor = processor
            self.model = model
            self.batch_size = config.get("batch_size")

        def training_step(self, batch, batch_idx):
            input_ids, attention_mask, pixel_values, image_sizes, labels = batch
            # print(f"Input IDs shape: {input_ids.shape}")
            # print(f"Attention mask shape: {attention_mask.shape}")
            # print(f"Pixel values shape: {pixel_values.shape}")
            # print(f"Image sizes shape: {image_sizes.shape}")
            # print(f"Labels shape: {labels.shape}")

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                labels=labels,
            )
            loss = outputs.loss

            self.log("train_loss", loss)

            return loss

        def validation_step(self, batch, batch_idx, dataset_idx=0):
            input_ids, attention_mask, pixel_values, image_sizes, answers = batch

            # Generate predictions
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                max_new_tokens=MAX_LENGTH,
            )

            # Decode predictions
            predictions = self.processor.batch_decode(
                generated_ids[:, input_ids.size(1) :], skip_special_tokens=True
            )

            # Calculate edit distance
            scores = []
            for pred, answer in zip(predictions, answers):
                pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
                scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

                if self.config.get("verbose", False) and len(scores) == 1:
                    print(f"Prediction: {pred}")
                    print(f"    Answer: {answer}")
                    print(f" Normed ED: {scores[0]}")

            # Log validation metric
            self.log("val_edit_distance", np.mean(scores))

            return scores

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.get("lr"))
            return optimizer

        def train_dataloader(self):
            return DataLoader(
                train_dataset,
                collate_fn=train_collate_fn,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
            )

        def val_dataloader(self):
            return DataLoader(
                val_dataset,
                collate_fn=eval_collate_fn,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
            )
    
    config = {
        "max_epochs": 1,
        # "val_check_interval": 0.2, # how many times we want to validate during an epoch
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 1,
        "lr": 2e-4,
        "batch_size": 4,
        # "seed":2022,
        "num_nodes": 1,
        "warmup_steps": 50,
        "result_path": "./result",
        "verbose": True,
    }
            
    return LlavaModelPLModule(config, processor, model), config
