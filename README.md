# LLaVa-NeXT for X-ray Radiology Report Generation

This repository contains the implementation and evaluation of the **LLaVa-NeXT** model for generating structured reports from medical X-ray images. The model has been adapted to process multiple images per patient and fine-tuned for specific anatomical regions in radiology reports.

# setup

clone the following resporitory 

**git clone https://github.com/oibrahi1/LLaVa-NEXT-Finetuning.git**

environmentset up 

**chmod 777 install.sh**

**./install.sh**

# preprocessing

The reports that are not sepertaed into the 4 anatomical regions, can be passed to this function to use prompt engineering with gpt-4 to categorize the report to the required parts. You will need an API Key to be passed to the model. 

**python src/preprocess_datasets.py --api_key <API_KEY>**

# training

After preprocessing all the data in train, test, and val to have categorized reports, this preprossed data is passed to train.py for fine tun ing using QLoRa with 4-bit quantization. you will be asked to pass your preprocessed data so it is expected to be in the main directory along with the images folder that has subfolder with images for each patient. 

**python train.py**

**Enter your preprocessed data (.json):processed.json**

If you want to fine tune with a subset of the data, you can do so **Subset? [y/n]**

You can save the fine tuned model in the model directory to be used for evaluation and inference. My fine tuned model is already saved in model-LLaVaNext

**Please specify save directory: model**

# Generate Reports

The fine-tuned model can be saved, and loaded later in the inference process to generate report for patients based on their X-Ray images. You will need to pass your preprossed data (.json) so it can be used to generate reports using the images only. 

**python src/generate_reports.py**

**Enter your preprocessed data (.json):processed.json**

To load your saved model 

**Model directory: model**

The genrated reports and the ground truth reports will be saved in the results directory to be used in the evaluation step. 

# Compute green score

After generating reports for testing and validation datsets, the generated reports and compared to the ground truth using the GREEN score, the average GREEN score for each anatomical region is computed and displayed on the screen and the values for each patient are saved in the results directory (green_scores). 

**python src/evaluate.py**

# Acknowledgments
This project builds upon the LLaVa-NeXT model and incorporates advanced techniques like LoRA and QLoRA for efficient fine-tuning. Part of the scripts in this resporitory were adapted from NielsRogge from his work on LLaVA NEXT fine-tuning. 

