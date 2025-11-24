#!/bin/bash

mkdir -p data
git clone https://huggingface.co/datasets/google/frames-benchmark data/frames
cd data/frames && git pull origin refs/pr/20
