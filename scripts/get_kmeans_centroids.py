"""
Get KMeans centroids for a given dataset.
"""

import argparse
import os

import faiss
import numpy as np
import pandas as pd
from numpy.linalg import norm


def get_args():
    parser = argparse.ArgumentParser(description="KMeans Centroid Calculation")
    parser.add_argument(
        "--nlist",
        type=int,
        default=256,
        help="Number of clusters for KMeans",
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default="/tmp/vectordb_bench/dataset/openai/openai_medium_500k",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="/tmp/vectordb_bench/centroids/kmeans-centroids/openai_medium_500k",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=512,
        help="Dimension of the embeddings.",
    )
    return parser.parse_args()


def load_dataset(file_path):
    print("Loading dataset from:", file_path)

    # load parquet files
    train_vectors = pd.read_parquet(f"{file_path}/train.parquet")

    # sort by id
    train_vectors.sort_values(by="id", inplace=True)
    train_ids = train_vectors["id"].to_numpy(dtype=np.int64)
    train_vectors = np.vstack(train_vectors["emb"].values)
    train_vectors /= norm(train_vectors, axis=1, keepdims=True)
    print(f"train_vectors shape: {train_vectors.shape}")

    return train_vectors.astype(np.float32)


def main():
    args = get_args()

    nlist = args.nlist
    seed = 42

    # prepare dataset
    train_vectors = load_dataset(args.file_path)
    dim = train_vectors.shape[1]
    assert dim == args.dim, f"Expected dimension {args.dim}, but got {dim}"
    print("✅ Load dataset complete.")

    # kmeans using faiss
    kmeans = faiss.Kmeans(dim, nlist, niter=25, seed=seed, verbose=True, gpu=True)
    kmeans.train(train_vectors)
    print("✅ KMeans training complete.")

    # allocate
    _, labels = kmeans.index.search(train_vectors, 1)
    labels = labels.flatten()
    centroids = kmeans.centroids
    print(f"Labels shape: {labels.shape}")
    print(f"Centroids shape: {centroids.shape}")

    # normalize
    centroids /= norm(centroids, axis=1, keepdims=True)
    print(f"Norm: {norm(centroids, axis=1)}")

    # save centroids
    os.makedirs(args.out_path, exist_ok=True)
    file_name = os.path.join(args.out_path, f"centroids_{nlist}.npy")
    np.save(file_name, centroids)
    print(f"✅ Centroids saved to {file_name}")


if __name__ == "__main__":
    main()
