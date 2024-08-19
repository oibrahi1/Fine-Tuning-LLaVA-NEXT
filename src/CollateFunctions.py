from .settings import *


def get_collate_funcs(processor):

    def verify_image_token_count(input_ids, num_images):
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        num_image_tokens = sum(
            1 for token in input_ids[0].cpu().numpy() if token == image_token_id
        )
        print(f"Number of <image> tokens: {num_image_tokens}")
        print(f"Expected number of images: {num_images}")
        assert (
            num_image_tokens == num_images
        ), f"Mismatch between image tokens ({num_image_tokens}) and number of images ({num_images})."


    def train_collate_fn(examples):
        all_images = []
        texts = []

        for example in examples:
            images, ground_truth = example
            all_images.extend(images)
            image_tokens = "".join(["<image>" for _ in images])

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract JSON"},
                        {"type": "text", "text": image_tokens},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ground_truth},
                    ],
                },
            ]
            text_prompt = processor.apply_chat_template(conversation)
            texts.append(text_prompt)

        batch = processor(
            text=texts,
            images=all_images,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return (
            batch["input_ids"],
            batch["attention_mask"],
            batch["pixel_values"],
            batch["image_sizes"],
            batch["labels"],
        )


    def eval_collate_fn(examples):
        images = []
        texts = []
        answers = []
        for example in examples:
            image, ground_truth = example
            images.extend(image)  # Extend the list by all images
            image_tokens = "".join(
                ["<image>" for _ in image]
            )  # Create an image token for each image

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract JSON"},
                        {"type": "text", "text": image_tokens},
                    ],
                },
            ]
            text_prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            texts.append(text_prompt)
            answers.append(ground_truth)

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # Verify the number of image tokens matches the number of images
        verify_image_token_count(batch["input_ids"], len(images))

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]
        image_sizes = batch["image_sizes"]

        return input_ids, attention_mask, pixel_values, image_sizes, answers
    
    return train_collate_fn, eval_collate_fn
