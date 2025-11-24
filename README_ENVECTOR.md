# enVector with ANN (IVF-GAS) in VectorDBBench

This guide demonstrates how to use enVector with an ANN index in VectorDBBench.

Basic usage of enVector with VectorDBBench follows the standard procedure for [VectorDBBench](https://github.com/zilliztech/VectorDBBench).

## Structure

```bash
.
├── centroids
│   └── embeddinggemma-300m
│       ├── centroids.npy             # centroids file for ANN
│       └── tree_info.pkl             # tree metadata for ANN
├── dataset
│   └── pubmed768d400k                # VectorDB ANN benchmark dataset
│       ├── neighbors.parquet
│       ├── test.npy
│       └── train.pkl
├── README_ENVECTOR.md
├── scripts
    ├── run_benchmark.sh              # benchmark script
    ├── envector_pubmed_config.yml    # benchmark config file
    └── prepare_dataset.py            # download and prepare ground truth neighbors for dataset
```

## Prerequisites

### Install Python Dependencies
```bash
# 1. Create your environment
python -m venv .venv
source .venv/bin/activate

# 2. Install VectorDBBench
pip install -e .

# 3. Install es2
pip install es2==1.2.0a4
```

### Prepare dataset

Prepare the following artifacts for the ANN benchmark with `scripts/prepare_dataset.py`:

- download datasets from HuggingFace
- prepare ground-truth neighbors
- download centroids and tree metadata for the GAS index for corresponding to the embedding model

For the ANN benchmark, we provide two datasets via HuggingFace:
- PUBMED768D400K: [cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m](https://huggingface.co/datasets/cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m)
- BLOOMBERG768D368K: [cryptolab-playground/Bloomberg-Financial-News-embedding-gemma-300m](https://huggingface.co/datasets/cryptolab-playground/Bloomberg-Financial-News-embedding-gemma-300m)

Also, we provide centroids and tree metadata for the corresponding embedding model used in the ANN benchmark:
- GAS Centroids: [cryptolab-playground/gas-centroids](https://huggingface.co/datasets/cryptolab-playground/gas-centroids)

To prepare dataset, run the following command as example:

```bash
# Prepare dataset
python ./scripts/prepare_dataset.py \
    -d cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m \
    -e embeddinggemma-300m
```

Then, you can find the generated files as follow:

```bash
.
├── centroids
│   └── embeddinggemma-300m
│       ├── centroids.npy
│       └── tree_info.pkl
└── dataset
    └── pubmed768d400k
        ├── neighbors.parquet
        ├── test.npy
        └── train.pkl
```

### Prepare enVector Server

To run enVector server with ANN, please refer to the [enVector Deployment repository](https://github.com/CryptoLabInc/envector-deployment). 
For example, you can start the server with the following command:

```bash
# Start enVector server
git clone https://github.com/CryptoLabInc/envector-deployment
cd envector-deployment/docker-compose
./start_envector.sh
```

We provide four enVector Docker Images:
- `cryptolabinc/es2e:v1.2.0-alpha.4`
- `cryptolabinc/es2b:v1.2.0-alpha.4`
- `cryptolabinc/es2o:v1.2.0-alpha.4`
- `cryptolabinc/es2c:v1.2.0-alpha.4`

### Set Environment Variables

```bash
# Set environment variables
export DATASET_LOCAL_DIR="./dataset"
export NUM_PER_BATCH=4096
```

## Run Our Benchmarks

We provide two benchmark datasets:
- `PUBMED768D400K`
- `BLOOMBERG768D368K`

Run the provided shell scripts (`./scripts/run_benchmark.sh`) as the following:

```bash
./scripts/run_benchmark.sh --type flat        # FLAT
./scripts/run_benchmark.sh --type ivf         # IVF-FLAT with random centroids
./scripts/run_benchmark.sh --type ivf-trained # IVF-FLAT with trained centroids (w/ k-means clustering, etc.)
./scripts/run_benchmark.sh --type ivf-gas     # IVF-FLAT with enVector-customized ANN (GAS)
```

For more details, please refer to `run_benchmark.sh` or `envector_{benchmark}_config.yml` in scripts directory for benchmarks with enVector with ANN (VCT), or you can use the following command:

```bash
# ivf-gas: IVF-FLAT with our ANN (GAS)
export NUM_PER_BATCH=500000 # set to the database size for efficiency
python -m vectordb_bench.cli.vectordbbench envectorivfflat \
    --uri "localhost:50050" \
    --case-type "PerformanceCustomDataset" \
    --db-label "PUBMED768D400K-IVF" \
    --custom-case-name PUBMED768D400K \
    --custom-dataset-name PUBMED768D400K \
    --custom-dataset-dir "" \
    --custom-dataset-size 400335 \
    --custom-dataset-dim 768 \
    --custom-dataset-file-count 1 \
    --custom-dataset-with-gt \
    --skip-custom-dataset-use-shuffled \
    --eval-mode mm \
    --train-centroids True \
    --is-vct True \
    --centroids-path "./centroids/embeddinggemma-300m/centroids.npy" \
    --vct-path "./centroids/embeddinggemma-300m/tree_info.pkl" \
    --nlist 32768 \
    --nprobe 6
```

### Run VectorDBBench Case

```bash
# flat
python -m vectordb_bench.cli.vectordbbench envectorflat \
    --uri "localhost:50050" \
    --case-type "Performance1536D500K" \
    --db-label "Performance1536D500K-FLAT"

# ivf: IVF-FLAT with random centroids
export NUM_PER_BATCH=500000  # set database size for efficiency
python -m vectordb_bench.cli.vectordbbench envectorivfflat \
    --uri "localhost:50050" \
    --case-type "Performance1536D500K" \
    --db-label "Performance1536D500K-IVF-FLAT" \
    --nlist 250 \
    --nprobe 6

# ivf-trained: IVF-FLAT with trained centroids via k-means
export NUM_PER_BATCH=500000 # set to the database size for efficiency
python -m vectordb_bench.cli.vectordbbench envectorivfflat \
    --uri "localhost:50050" \
    --case-type "Performance1536D500K" \
    --db-label "Performance1536D500K-IVF-FLAT" \
    --train-centroids True \
    --centroids-path "./centroids/kmeans_centroids.npy" \
    --nlist 250 \
    --nprobe 6
```

Note that the benchmark provided by VectorDBBench, including Performance1536D500K, uses **unknown** embedding model (just notified as openai's one), we cannot use our IVF-GAS approach for ANN.

### CLI Options

enVector Types for VectorDBBench
- `envectorflat`: FLAT as index type for enVector
- `envectorivfflat`: IVF_FLAT as index type for enVector

Common Options for enVector
- `--uri`: enVector server URI
- `--eval-mode`: FHE evaluation mode on server. Use `mm` for enhanced performance.

ANN Options for enVector
- `--nlist`: Number of coarse clusters for IVF_FLAT
- `--nprobe`: Number of clusters to scan during search for IVF_FLAT
- `--train-centroids`: whether to use trained centroids for IVF_FLAT
- `--centroids-path`: path to the trained centroids
- `--is-vct`: whether to use VCT approach for IVF_GAS
- `--vct-path`: path to the trained VCT metadata for IVF_GAS

Benchmark Options:
    follows conventions of VectorDBBench, 
    see details in [VectorDBBench Options](https://github.com/zilliztech/VectorDBBench?tab=readme-ov-file#custom-dataset-for-performance-case)