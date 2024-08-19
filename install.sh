#!/bin/bash

pip install -U bitsandbytes
pip uninstall bitsandbytes -y
pip install bitsandbytes
pip install -U transformers
pip install -U accelerate
pip install -U peft
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
pip install accelerate
pip install -U "transformers>=4.39.0"
pip install peft bitsandbytes
pip install -U "trl>=0.8.3"
pip install lightning
pip install datasets
pip install jinja2==3.1.4
pip install nltk==3.8.1
pip install openai==0.28.0
pip install python-dotenv

git clone https://github.com/Stanford-AIMI/GREEN.git
mv GREEN ./src
