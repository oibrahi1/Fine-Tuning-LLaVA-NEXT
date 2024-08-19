<<<<<<< HEAD
<<<<<<< HEAD
# setup
git clone https://github.com/oibrahi1/LLaVa-NEXT-Finetuning.git

chmod 777 install.sh

./install.sh

# preprocessing
python src/preprocess_datasets.py --api_key <API_KEY>

# training
python LLaVa_script.py

Enter your preprocessed data (.json):processed.json

Subset? [y/n]n

Please specify save directory:model

# Generate Reports
python src/generate_reports.py

Enter your preprocessed data (.json):processed.json

Model directory: model

# Compute green score
python src/evaluate.py
=======
# Fine-Tuning-LLaVA-NEXT
>>>>>>> cd5059eba450752beae1ca968df55404e0e03ca2
=======

>>>>>>> 2ef20af8cc90605e38273acf60151a804bd5036e
