import json
import torch
import os
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from pathlib import Path

# Assuming LlavaNextDatasetCustom is already defined in your project
from LlavaNextDatasetCustom import LlavaNextDatasetCustom
import re

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

# let's turn that into JSON
def token2json(tokens, processor, is_inner_value=False, added_vocab=None):
        """
        Convert a (generated) token sequence into an ordered JSON format.
        """
        if added_vocab is None:
            added_vocab = processor.tokenizer.get_added_vocab()

        output = {}

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            key_escaped = re.escape(key)

            end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(
                    f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE | re.DOTALL
                )
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = token2json(content, processor, is_inner_value=True, added_vocab=added_vocab)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + token2json(tokens[6:], processor, is_inner_value=True, added_vocab=added_vocab)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}


def load_model(local_model_dir):
    processor = AutoProcessor.from_pretrained(local_model_dir)
    with open(f"{local_model_dir}/quantization_config.json", "r") as f:
        quantization_config_dict = json.load(f)
    quantization_config = BitsAndBytesConfig(**quantization_config_dict)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        local_model_dir,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )
    return model, processor

def generate_reports(dataset_dict, model_dir, split="test"):
    model, processor = load_model(model_dir)

    # Initialize LlavaNextDatasetCustom datasets
    if split == "train":
        llava_dataset = LlavaNextDatasetCustom(dataset_dict, split="train")
    elif split == "validation":
        llava_dataset = LlavaNextDatasetCustom(dataset_dict, split="validation")
    elif split == "test":
        llava_dataset = LlavaNextDatasetCustom(dataset_dict, split="test")
    else:
        raise ValueError("Invalid split provided. Choose from 'train', 'validation', or 'test'.")
    llava_dataset = torch.utils.data.Subset(llava_dataset, range(20))

    # Initialize lists to store the generated JSON outputs and the corresponding ground truths
    generated_outputs = []
    ground_truths = []

    # Loop over the selected dataset split
    for idx, (images, ground_truth) in enumerate(llava_dataset):
        print(f"Processing example {idx + 1}/{len(llava_dataset)}")

        # Prepare the conversation with all images
        conversation_content = [{"type": "text", "text": "Extract JSON"}]
        for _ in images:
            conversation_content.append({"type": "image"})

        conversation = [
            {
                "role": "user",
                "content": conversation_content,
            },
        ]

        # Apply the chat template
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Prepare inputs for the model
        inputs = processor(text=text_prompt, images=images, return_tensors="pt").to("cuda")

        # Generate token IDs
        generated_ids = model.generate(**inputs, max_new_tokens=256)

        # Decode back into text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Convert generated text to JSON
        generated_json = token2json(generated_text, processor)

        # Save the generated JSON and the corresponding ground truth
        generated_outputs.append(generated_json)
        ground_truths.append(ground_truth)

    # Save the generated reports to the results directory
    output_path = Path("results") / f"generated_reports_{split}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as outfile:
        json.dump(generated_outputs, outfile, indent=4)
    output_path2 = Path("results") / f"ground_truths_{split}.json"
    output_path2.parent.mkdir(exist_ok=True)
    with open(output_path2, 'w') as outfile:
        json.dump(ground_truths, outfile, indent=4)
    print(f"Generated reports saved to {output_path}, ground truths saved to {output_path2}")

if __name__ == "__main__":
    # Assuming dataset_dict is loaded from a preprocessed dataset
    # dataset_dict = load_from_disk("results/datasetXray")
    dataset_dict = load_xray()
    model_dir = input("Model directory: ")
    
    # Choose the split you want to process: "train", "validation", or "test"
    generate_reports(dataset_dict, model_dir, split="test")