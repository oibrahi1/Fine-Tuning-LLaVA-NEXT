from torch.utils.data import Dataset
from typing import Any, Dict
from PIL import Image
from torchvision.transforms import Resize
import sys


def analyze_dataset(dataset, processor):
    max_image_size = 0
    max_token_length = 0
    total_images = 0
    total_tokens = 0

    for i, item in enumerate(dataset):
        images, target_sequence = item

        # Analyze images
        for img in images:
            total_images += 1
            img_size = sys.getsizeof(img.tobytes())
            max_image_size = max(max_image_size, img_size)

        # Analyze target sequence
        token_length = len(processor.tokenizer.encode(target_sequence))
        max_token_length = max(max_token_length, token_length)
        total_tokens += token_length

        if i % 100 == 0:
            print(f"Processed {i} items...")

        if i % 1000 == 999:
            print(f"Interim stats after {i+1} items:")
            print(f"Max image size: {max_image_size / 1024:.2f} KB")
            print(f"Max token length: {max_token_length}")
            print(f"Average tokens per item: {total_tokens / (i+1):.2f}")
            print(f"Average images per item: {total_images / (i+1):.2f}")
            print("---")

    print("\nFinal stats:")
    print(f"Total items: {len(dataset)}")
    print(f"Max image size: {max_image_size / 1024:.2f} KB")
    print(f"Max token length: {max_token_length}")
    print(f"Average tokens per item: {total_tokens / len(dataset):.2f}")
    print(f"Average images per item: {total_images / len(dataset):.2f}")


class LlavaNextDatasetCustom(Dataset):
    """
    PyTorch Dataset for LLaVa-NeXT adapted for your X-ray dataset.

    Each row consists of image paths and a ground truth report.
    """

    def __init__(self, dataset_dict, split: str = "train", sort_json_key: bool = True):
        super().__init__()

        self.split = split
        self.sort_json_key = sort_json_key

        self.dataset = dataset_dict[split]
        self.dataset_length = len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Convert a JSON object into a token sequence.
        """
        if isinstance(obj, dict):
            output = ""
            keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
            for k in keys:
                output += (
                    rf"<s_{k}>" + self.json2token(obj[k], sort_json_key) + rf"</s_{k}>"
                )
            return output
        elif isinstance(obj, list):
            return r"<sep/>".join(
                [self.json2token(item, sort_json_key) for item in obj]
            )
        else:
            return str(obj)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        sample = self.dataset[idx]
        images = [Image.open(img_path).convert("RGB") for img_path in sample["image"]]
        images = [Resize((224, 224))(img) for img in images]  # Resize to a smaller size
        ground_truth = sample["ground_truth"]
        target_sequence = self.json2token(
            ground_truth, sort_json_key=self.sort_json_key
        )
        return images, target_sequence
