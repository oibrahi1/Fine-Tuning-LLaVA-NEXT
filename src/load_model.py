import json
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch
import os

def load_model(local_model_dir):
    # Load the processor
    processor = AutoProcessor.from_pretrained(local_model_dir)

    # Load the quantization config from the JSON file
    with open(os.path.join(local_model_dir, "quantization_config.json"), "r") as f:
        quantization_config_dict = json.load(f)

    # Create a BitsAndBytesConfig object from the loaded dictionary
    quantization_config = BitsAndBytesConfig(**quantization_config_dict)

    # Load the model with the quantization config
    model = LlavaNextForConditionalGeneration.from_pretrained(
        local_model_dir,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )

    return model, processor

if __name__ == "__main__":
    local_model_dir = "./model"
    model, processor = load_model(local_model_dir)
    print("Model, processor, and quantization config loaded successfully!")
