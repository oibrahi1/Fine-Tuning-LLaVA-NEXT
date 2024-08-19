import os
import json
import argparse
import openai
from dotenv import load_dotenv

def process_report_with_gpt(report, api_key):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": f"""You are given a radiology report of a chest X-Ray. Your task is to separate the findings into four predefined anatomical regions: lung, heart, mediastinal, and bone. If you cannot assign a sentence to any anatomical region, put it in 'others'. Here is an example:

Input: The cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size. The lungs are mildly hypoinflated but grossly clear of focal airspace disease, pneumothorax, or pleural effusion. There are mild degenerative endplate changes in the thoracic spine. There are no acute bony findings.

Expected output:
{{
    "lung": "Lungs are mildly hypoinflated but grossly clear of focal airspace disease, pneumothorax, or pleural effusion. Pulmonary vasculature are within normal limits in size.",
    "heart": "Cardiac silhouette within normal limits in size.",
    "mediastinal": "Mediastinal contours within normal limits in size.",
    "bone": "Mild degenerative endplate changes in the thoracic spine. No acute bony findings.",
    "others": ""
}}

Now, process the following report: {{"original_report": "{report}"}}"""
            }
        ]
    )
    return response['choices'][0]['message']['content']

def main():
    parser = argparse.ArgumentParser(description="Preprocess radiology reports by separating findings into different anatomical regions.")
    parser.add_argument('--api_key', type=str, required=True, help="OpenAI API key for accessing GPT-4.")
    parser.add_argument('--input_file', type=str, required=False, help="Path to the input JSON file containing the reports.", default="annotation_quiz_all.json")
    parser.add_argument('--output_file', type=str, required=False, help="Path to save the processed JSON file.", default="processed.json")
    parser.add_argument('--split', type=str, required=False, choices=['train', 'val', 'test'], help="Which split of the dataset to process.", default='val')

    args = parser.parse_args()

    # Load the input data
    with open(args.input_file, 'r') as infile:
        data = json.load(infile)

    # Process the specified split
    processed_split = []
    for item in data[args.split]:
        original_report = item.pop('original_report')
        processed_report = process_report_with_gpt(original_report, args.api_key)
        item['report'] = json.loads(processed_report)
        processed_split.append(item)

    # Save the processed data
    data[args.split] = processed_split
    with open(args.output_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    print(f"Processed '{args.split}' data saved to '{args.output_file}'")

if __name__ == "__main__":
    main()
