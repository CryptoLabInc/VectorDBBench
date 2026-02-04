"""
Prepare dataset and ground truth neighbors for benchmarking.
"""

import argparse
import os

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import wget
from datasets import load_dataset


SUPPORTED_CASES = {
    "pubmed768d400k": {
        "dataset_name": "cryptolab-playground/pubmed-arxiv-abstract-embedding-gemma-300m",
        "embedding_model": "embeddinggemma-300m"
    },
    "bloomberg768d368k": {
        "dataset_name": "cryptolab-playground/Bloomberg-Financial-News-embedding-gemma-300m",
        "embedding_model": "embeddinggemma-300m"
    },
    "products512d400k": {
        "dataset_name": "cryptolab-playground/amazon-products-clip-vit-b-32",
        "embedding_model": "clip-vit-b-32"
    },
    "food512d101k": {
        "dataset_name": "cryptolab-playground/food101-clip-vit-b-32",
        "embedding_model": "clip-vit-b-32"
    }
}
SUPPORTED_EMBEDDING_MODELS = ["embeddinggemma-300m", "clip-vit-b-32"]


def get_args():
    parser = argparse.ArgumentParser(description="Prepare dataset and ground truth neighbors for benchmarking.")
    parser.add_argument(
        "-d",
        "--dataset-name",
        type=str,
        default="pubmed768d400k",
        help="Huggingface dataset name to download.",
        choices=list(SUPPORTED_CASES.keys()),
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory to save the dataset and neighbors. Default: <dataset_name> in DATASET_LOCAL_DIR.",
    )
    parser.add_argument(
        "--centroids-dir",
        type=str,
        default="./centroids",
        help="Directory to save the centroids.",
    )
    return parser.parse_args()


def download_dataset(dataset_name: str, output_dir: str = "./dataset/pubmed768d400k") -> None:
    """Download dataset from Huggingface and save as Parquet files."""
    # load dataset
    ds = load_dataset(SUPPORTED_CASES[dataset_name]["dataset_name"])
    train = ds["train"].to_pandas()
    test = ds["test"].to_pandas()

    # write to parquet
    train_table = pa.Table.from_pandas(train)
    pq.write_table(train_table, f"{output_dir}/train.parquet")

    test_table = pa.Table.from_pandas(test)
    pq.write_table(test_table, f"{output_dir}/test.parquet")


def prepare_neighbors(
    data_dir: str = "./dataset/pubmed768d400k",
) -> None:
    """Prepare ground truth neighbors using brute-force flat search and save as Parquet."""
    # load dataset
    train = pd.read_parquet(f"{data_dir}/train.parquet")
    test = pd.read_parquet(f"{data_dir}/test.parquet")

    train = np.stack(train["emb"].to_list()).astype("float32")
    test = np.stack(test["emb"].to_list()).astype("float32")
    dim = train.shape[1]

    # flat search
    index = faiss.IndexFlatIP(dim)
    index.add(train)

    k = len(test)
    distances, indices = index.search(test, k)
    print(distances.shape, indices.shape)

    # save flat search result as neighbors
    df = pd.DataFrame({"id": np.arange(len(indices)), "neighbors_id": indices.tolist()})

    table = pa.Table.from_pandas(df)
    pq.write_table(table, f"{data_dir}/neighbors.parquet")


def download_centroids(embedding_model: str, dataset_dir: str) -> None:
    """Download pre-computed centroids and for IVF_GAS index."""

    if embedding_model not in SUPPORTED_EMBEDDING_MODELS:
        raise ValueError(f"Centroids for {embedding_model} currently not available.")

    # BASE URL: https://huggingface.co/datasets/cryptolab-playground/gas-centroids
    dataset_link = (
        f"https://huggingface.co/datasets/cryptolab-playground/gas-centroids/resolve/add-dataset-v1/{embedding_model}"
    )

    # download
    os.makedirs(os.path.join(dataset_dir, embedding_model), exist_ok=True)
    wget.download(f"{dataset_link}/centroids.npy", out=os.path.join(dataset_dir, embedding_model, "centroids.npy"))
    print(f"\nDownloaded centroids to {os.path.join(dataset_dir, embedding_model)}")


if __name__ == "__main__":
    args = get_args()

    base_dataset_dir = os.environ.get("DATASET_LOCAL_DIR", "/tmp/vectordb_bench/dataset") if args.dataset_dir is None else args.dataset_dir
    args.dataset_dir = os.path.join(base_dataset_dir, args.dataset_name)
    os.makedirs(args.dataset_dir, exist_ok=True)

    download_dataset(args.dataset_name, args.dataset_dir)
    prepare_neighbors(args.dataset_dir)
    download_centroids(SUPPORTED_CASES[args.dataset_name]["embedding_model"], args.centroids_dir)
