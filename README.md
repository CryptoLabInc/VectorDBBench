# enVector in VectorDBBench

This guide demonstrates how to use [enVector](https://docs.envector.io/) in VectorDBBench.
The **enVector** is a vector search engine that lets you search directly on encrypted data.
VectorDBBench with enVector provides toolkits to measure and compare performance across different index types.

Basic usage of enVector with VectorDBBench follows the standard procedure for [VectorDBBench](https://github.com/zilliztech/VectorDBBench).

## 🚀 Quick Start

1. Run enVector server

```bash
# Start enVector server
git clone https://github.com/CryptoLabInc/envector-deployment
cd envector-deployment/docker-compose
./start_envector.sh
```

2. Install Python Dependencies

```bash
# Install Python Dependencies
pip install -e .
pip install pyenvector==1.3.0a1
```

3. Run Benchmark

```bash
# Run Benchmark (VectorDBBench built-in dataset)
./scripts/run_benchmark.sh --index-type FLAT --config-file envector_openai_config.yml
```

## 📚 Core Concepts

### Index Types

- FLAT: brute-force search.
- IVF-FLAT: Inverted file (IVF) index with flat vectors for faster approximate search. Searchs only the nearest clusters with trained centroids instead of the entire database.
- IVF-GAS: enVector-customized ANN algorithm for the fastest approximate search. To test this, we provided benchmark datasets.

| Index Type | Speed | Accuracy | Centroids Setup | Dataset Requirements |
|------------|-------|----------|------------------|---------------------|
| FLAT | Slow | 100% | None | Any dataset |
| IVF-FLAT | Fast | 95-99% | Requires to train k-means centroids with the client's dataset | Any dataset |
| IVF-GAS | Fastest* | 95-99% | Requires the provided centroids (see [Prepare Dataset](#prepare-dataset)) | enVector custom datasets only |

### Benchmark Cases

enVector supports two types of benchmark cases:

| Benchmark Type | Description | Dataset Preparation | Available Index Types |
|----------------|-------------|---------------------|----------------------|
| **VectorDBBench Built-in** | Standard benchmarks from VectorDBBench | Not required (auto-downloaded) | FLAT, IVF-FLAT |
| **enVector Custom Cases** | Optimized for encrypted search with GAS | Required (see [Prepare Dataset](#prepare-dataset)) | FLAT, IVF-FLAT, IVF-GAS |


## 📁 Project Structure

```bash
.
├── README.md
├── scripts
│   ├── get_kmeans_centroids.py              # create kmeans centroids
│   ├── requirements.txt                     # python requirements
│   ├── prepare_dataset.py                   # download and prepare ground truth neighbors for dataset
│   └── run_benchmark.sh                     # benchmark script
└── vectordb_bench/config-files              # benchmark config file
    └── envector_{benchmark_case}_config.yml
```

## 🔧 Prerequisites

### 1. Install Python Dependencies
```bash
# 1. Create your environment
python -m venv .venv
source .venv/bin/activate

# 2. Install VectorDBBench
pip install -e .

# 3. Install pyenvector
# pip uninstall pyenvector  # if installed
pip install pyenvector==1.3.0a1
```

### 2. Prepare enVector Server

To run enVector server with ANN, please refer to the [enVector Deployment repository](https://github.com/CryptoLabInc/envector-deployment). 
For example, you can start the server with the following command:

```bash
# Start enVector server
git clone https://github.com/CryptoLabInc/envector-deployment
cd envector-deployment/docker-compose
./start_envector.sh
```

We provide 5 enVector Docker Images:
- `cryptolabinc/envector-endpoint:v1.3.0-alpha.1`
- `cryptolabinc/envector-backend:v1.3.0-alpha.1`
- `cryptolabinc/envector-shaper:v1.3.0-alpha.1`
- `cryptolabinc/envector-orchestrator:v1.3.0-alpha.1`
- `cryptolabinc/envector-compute:v1.3.0-alpha.1`

## 📊 Run Benchmark

### 1. VectorDBBench Built-in Cases

Run the following commands to run enVector with VectorDBBench's built-in benchmark.

```bash
./scripts/run_benchmark.sh --index-type FLAT --config-file envector_{benchmark_case}_config.yml # FLAT
./scripts/run_benchmark.sh --index-type IVF_FLAT  --config-file envector_{benchmark_case}_config.yml # IVF-FLAT
```

For more details, please refer to `envector_{benchmark_case}_config.yml` in scripts directory for benchmarks with enVector, or you can use the following command:

```bash
python -m vectordb_bench.cli.vectordbbench envectorflat \
    --config-file envector_openai_config.yml

# or

python -m vectordb_bench.cli.vectordbbench envectorflat \
    --uri "localhost:50050" \
    --case-type "Performance1536D500K"
```

If you need the trained k-means centroids, run `./scripts/get_kmeans_centroids.py` with your dataset.

Note that, the benchmark provided by VectorDBBench, including Performance1536D500K, uses **unknown** embedding model (just notified as openai's one), we cannot use our GAS approach for ANN.


### 2. enVector Custom Cases (with GAS Support)

We provide enVector-customized ANN, called "GAS", designed to perform efficient IVF-FLAT-based ANN search with the encrypted index.
We evaluated enVector on benchmark datasets that we provided.

#### 2-1. Prepare dataset

Prepare the following artifacts for the ANN benchmark with `scripts/prepare_dataset.py`:

- download datasets from HuggingFace
- prepare ground-truth neighbors
- download centroids for the GAS index for corresponding to the embedding model

For the ANN benchmark, we provide two datasets via HuggingFace:
- `PUBMED768D400K`: [cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m](https://huggingface.co/datasets/cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m)
- `BLOOMBERG768D368K`: [cryptolab-playground/Bloomberg-Financial-News-embedding-gemma-300m](https://huggingface.co/datasets/cryptolab-playground/Bloomberg-Financial-News-embedding-gemma-300m)
- `PRODUCTS512D400K`: [cryptolab-playground/amazon-products-clip-vit-b-32](https://huggingface.co/datasets/cryptolab-playground/amazon-products-clip-vit-b-32)

Also, we provide centroids for the corresponding embedding model used in the ANN benchmark:
- GAS Centroids: [cryptolab-playground/gas-centroids](https://huggingface.co/datasets/cryptolab-playground/gas-centroids)

To prepare dataset, run the following command as example:

```bash
# Install dependencies for preparing dataset
pip install -r ./scripts/requirements.txt

# Prepare GAS dataset: PUBMED768D400K
python ./scripts/prepare_dataset.py \
    -d cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m \
    -e embeddinggemma-300m

# Prepare GAS dataset: BLOOMBERG768D368K
python ./scripts/prepare_dataset.py \
    -d cryptolab-playground/Bloomberg-Financial-News-embedding-gemma-300m \
    -e embeddinggemma-300m

# Prepare GAS dataset: PRODUCTS512D400K
python ./scripts/prepare_dataset.py \
    -d playground/amazon-products-clip-vit-b-32 \
    -e clip-vit-b-32
```

Then, you can find the generated files as follows:

```bash
.
├── centroids                                # centroids for IVF index types
│   └── embeddinggemma-300m                  # centroids for IVF-GAS
│       └── centroids.npy
└── dataset                                  # custom benchmark dataset
    └── pubmed768d400k
        ├── neighbors.parquet
        ├── test.parquet
        └── train.parquet
```

#### 2-2. Run enVector Custom Cases

Run the provided shell scripts (`./scripts/run_benchmark.sh`) as the following:

```bash
./scripts/run_benchmark.sh --index-type FLAT --config-file envector_pubmed_config.yml     # FLAT
./scripts/run_benchmark.sh --index-type IVF_FLAT --config-file envector_pubmed_config.yml # IVF-FLAT with trained k-means centroids
./scripts/run_benchmark.sh --index-type IVF_GAS --config-file envector_pubmed_config.yml  # GAS: enVector-customized ANN
```

For more details, please refer to `run_benchmark.sh` or `envector_{benchmark_case}_config.yml` in scripts directory for benchmarks with enVector with ANN (GAS), or you can use the following command:

```bash
python -m vectordb_bench.cli.vectordbbench envectorivfflat \
    --config-file envector_pubmed_config.yml

# or 

python -m vectordb_bench.cli.vectordbbench envectorivfflat \
    --uri "localhost:50050" \
    --eval-mode mm \
    ... \
    --train-centroids True \
    --centroids-path "./centroids/embeddinggemma-300m/centroids.npy" \
    --nlist 1024 \
    --nprobe 6
```

Note that, **`NUM_PER_BATCH` should be set to the database size** when using IVF-based ANN index for enVector currently.
We will support adjustable `NUM_PER_BATCH` for ANN soon.

## 🎯 Advanced Usage

### Prepare Other Datasets

If you want to test on other benchmark datasets regardless ANN benchmark, please run the following scripts:

```python
# (Optional) Prepare random dataset
python ./scripts/prepare_random_dataset.py \
    --dataset-dir ./dataset/random512d1m \
    --dataset-size 1_000_000
```

### enVector VectorDBBench CLI Options

enVector Types for VectorDBBench
- `envectorflat`: FLAT index type for enVector
- `envectorivfflat`: IVF_FLAT index type for enVector
- `envectorivfgas`: GAS index type for enVector

Common Options for enVector
- `--uri`: enVector server URI
- `--eval-mode`: FHE evaluation mode on server. Use `mm` for enhanced performance.

ANN Options for enVector
- `--nlist`: Number of coarse clusters for IVF index types.
- `--nprobe`: Number of clusters to scan during search for IVF index types.
- `--train-centroids`: Whether to use trained centroids for IVF index types. Default is False, which means to use randomly generated centroids.
- `--centroids-path`: Path to the trained centroids for IVF index types.

Benchmark Options:
follows conventions of VectorDBBench, 
see details in [VectorDBBench Options](https://github.com/zilliztech/VectorDBBench?tab=readme-ov-file#custom-dataset-for-performance-case).
For example, if you have a custom directory for dataset, set `DATASET_LOCAL_DIR`.

### enVector VectorDBBench Config File Options

You can file the customized config files in `vectordb_bench/config-files` to use CLI options in more convinient way.

```yaml
# FLAT
envectorflat:
  index_name: test_index
  uri: localhost:50050
  eval_mode: mm
  case_type: Performance1536D500K
  db_label: Performance1536D500K-FLAT
  k: 10
  drop_old: true
  load: true

# IVF-FLAT with trained k-means centroids
envectorivfflat:
  ...
  nlist: 256
  nprobe: 6
  train_centroids: true
  centroids_path: centroids/performance1536d500k/centroids_256.npy

# GAS: enVector-customized ANN
envectorivfgas:
  ...
```


## ❓ Troubleshooting

- `RuntimeError: Failed to connect to localhost:50050`: Required to run enVector server.
- `polars.exceptions.ColumnNotFoundError: "emb" not found`: Required to provide the correct dataset path to env var `DATASET_LOCAL_DIR`.