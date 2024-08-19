import os
import gc
import json
import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from src.PyTorchDataset import LlavaNextDatasetCustom, analyze_dataset
from src.CollateFunctions import get_collate_funcs
from src.LightningModule import get_LlavaModelPLModule, get_callback
from src.settings import *


def log_gpu_memory():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    
class MemoryLoggingCallback(L.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:  # Log every 100 batches
            log_gpu_memory()
            gc.collect()
            torch.cuda.empty_cache()


def load_xray():
    # Paths to your data
    data_dir = "./"
    images_dir = os.path.join(data_dir, "images")
    json_file = os.path.join(data_dir, input("Enter your preprocessed data (.json):"))

    # Load the JSON file
    with open(json_file, "r") as file:
        data = json.load(file)


    def load_data(data_split):
        
        images_list = []
        reports_list = []

        for item in data[data_split]:
            patient_id = item["id"]
            report = item["report"]  # Keep the full report, including "others"

            # Construct the path to the patient's images
            patient_image_dir = os.path.join(images_dir, patient_id)

            # Collect all images for this patient
            images = []
            for image_name in sorted(os.listdir(patient_image_dir)):
                if image_name.endswith(".png"):  # Ensure it's a PNG image
                    image_path = os.path.join(patient_image_dir, image_name)
                    images.append(image_path)

            # Append the images and report to the respective lists
            images_list.append(images)
            reports_list.append(report)

        return {"image": images_list, "ground_truth": reports_list}


    # Load data for each split
    train_data = load_data("train")
    val_data = load_data("val")
    test_data = load_data("test")

    # Create Hugging Face datasets
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)
    test_dataset = Dataset.from_dict(test_data)

    datasetXray = DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    )
    return datasetXray


def load_model():
    ## Load model

    # Three options for training, from the lowest precision training to the highest precision training:
    # - QLora
    # - Standard Lora
    # - Full fine-tuning
    bnb_config = None
    if USE_QLORA or USE_LORA:
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )
    else:
        # for full fine-tuning, we can speed up the model using Flash Attention
        # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
        )
    return model, bnb_config


def apply_peft(model):
    
    def find_all_linear_names(model):
        cls = torch.nn.Linear
        lora_module_names = set()
        multimodal_keywords = ["multi_modal_projector", "vision_model"]
        for name, module in model.named_modules():
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if "lm_head" in lora_module_names:  # needed for 16-bit
            lora_module_names.remove("lm_head")
        return list(lora_module_names)
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model


def save_model_processor_and_quant_config(model, processor, quantization_config, save_dir="model/llava_next_model"):
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    # Save the quantization config as JSON using the to_json_file method
    quantization_config.to_json_file(f"{save_dir}/quantization_config.json")

    print(f"Model, processor, and quantization config saved to {save_dir}")


if __name__ == '__main__':
    
    # load dataset
    datasetXray = load_xray()
    # datasetXray = load_from_disk('results/datasetXray')
    example = datasetXray["train"][0]
    print("Images:", example["image"])
    print("Ground Truth:", example["ground_truth"])
    
    # load processor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = (
        "right"  # during training, one always uses padding on the right
    )
    
    # load model and apply peft
    model, quant_config = load_model()
    model = apply_peft(model)
    
    # create PyTorch dataset
    train_dataset = LlavaNextDatasetCustom(datasetXray, split="train")
    val_dataset = LlavaNextDatasetCustom(datasetXray, split="validation")
    test_dataset = LlavaNextDatasetCustom(datasetXray, split="test")
    if input("Subset? [y/n]") == "y":
        train_dataset = torch.utils.data.Subset(train_dataset, range(20))
        val_dataset = torch.utils.data.Subset(val_dataset, range(5))
    train_example = train_dataset[0]
    images, target_sequence = train_example
    print(target_sequence)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print("Analyzing train dataset...")
    analyze_dataset(train_dataset, processor)
    print("\nAnalyzing validation dataset...")
    analyze_dataset(val_dataset, processor)
    
    # define collate functions and loaders
    train_collate_fn, eval_collate_fn = get_collate_funcs(processor)
    model_module, config = get_LlavaModelPLModule(train_dataset, val_dataset, train_collate_fn, eval_collate_fn, processor, model)
    train_loader = DataLoader(
        train_dataset, collate_fn=train_collate_fn, batch_size=1, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, collate_fn=eval_collate_fn, batch_size=1, shuffle=False
    )
    
    # define callbacks and train
    early_stop_callback = get_callback()
    csv_logger = CSVLogger(save_dir="logs", name="my_logs")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=4,
        check_val_every_n_epoch=1,
        gradient_clip_val=1,
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=0,
        logger=csv_logger,
        callbacks=[early_stop_callback, MemoryLoggingCallback()],
    )
    trainer.fit(model_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    model.merge_and_unload()
    save_model_processor_and_quant_config(model, processor, quant_config, input("Please specify save directory:"))
    