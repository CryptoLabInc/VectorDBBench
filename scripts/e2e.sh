#!/bin/bash

set -e

# Virtual Environment
python -m venv .venv
source .venv/bin/activate

# Install Python Dependencies
pip install -e .
pip install pyenvector==1.3.0a1
pip install -r scripts/requirements.txt

## Dataset Preparation
python ./scripts/prepare_random_dataset.py \
    --dataset-dir random512d10k \
    --dataset-size 10000

## Centroid Generation
python ./scripts/get_kmeans_centroids.py \
    --nlist 256 \
    --file-path random512d10k \
    --out-path centroids

## Benchmark Run
export NUM_PER_BATCH=4096

./scripts/run_benchmark.sh \
    --index-type FLAT \
    --config-file envector_sample_config.yml

./scripts/run_benchmark.sh \
    --index-type IVF_FLAT \
    --config-file envector_sample_config.yml

./scripts/run_benchmark.sh \
    --index-type IVF_VCT \
    --config-file envector_sample_config.yml