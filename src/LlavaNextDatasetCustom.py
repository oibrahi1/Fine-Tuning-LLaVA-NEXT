import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Resize

class LlavaNextDatasetCustom(Dataset):
    def __init__(self, dataset_dict, split="train"):
        self.dataset = dataset_dict[split]
        self.dataset_length = len(self.dataset)

    def json2token(self, obj):
        if isinstance(obj, dict):
            output = ""
            for k, v in obj.items():
                output += f"<s_{k}>{self.json2token(v)}</s_{k}>"
            return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item) for item in obj])
        else:
            return str(obj)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        images = [Image.open(img_path).convert("RGB") for img_path in sample["image"]]
        images = [Resize((224, 224))(img) for img in images]
        ground_truth = sample["ground_truth"]
        target_sequence = self.json2token(ground_truth)
        return images, target_sequence
