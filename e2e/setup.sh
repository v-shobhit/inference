#!/bin/bash

pip install -r requirements.txt
apt-get update
apt-get install -y --no-install-recommends \
    wkhtmltopdf \
    wget
python -m spacy download en_core_web_sm
