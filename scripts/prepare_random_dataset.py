"""
Prepare random dataset and ground truth neighbors for test purposes.
"""

import argparse
import os

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_args():
    parser = argparse.ArgumentParser(description="Prepare random dataset for benchmarking.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.join(os.environ.get("DATASET_LOCAL_DIR", "/tmp/vectordb_bench/dataset"), "random512d1m"),
        help="Directory to save the random vectors.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1_000_000,
        help="Number of dataset embeddings to use.",
    )
    parser.add_argument(
        "--query-size",
        type=int,
        default=1_000,
        help="Number of query embeddings to use. 1,000 is recommended in VectorDBBench.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=512,
        help="Dimension of the embeddings.",
    )

    return parser.parse_args()


def get_random_data(num_data, dim, seed):
    rng = np.random.default_rng(seed)

    data = rng.uniform(low=-1.0, high=1.0, size=(num_data, dim))

    # L2 normalize
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-10)
    data /= norm

    print(data.shape)
    return data.astype(np.float32)


def npy_to_parquet(
    vector: np.ndarray,
    dataset_dir: str = "./dataset/random512d1m",
    mode: str = "train",
) -> None:
    """Convert downloaded .npy embeddings to Parquet format."""
    print("Preparing embeddings from numpy array...")
    os.makedirs(dataset_dir, exist_ok=True)

    ids = np.arange(len(vector))
    id_array = pa.array(ids, type=pa.int64())

    list_arrays = [vector[i].tolist() for i in range(len(vector))]
    vector_array = pa.array(list_arrays, type=pa.list_(pa.float64()))

    assert len(id_array) == len(vector_array)

    table = pa.Table.from_arrays([id_array, vector_array], names=["id", "emb"])

    out_path = os.path.join(dataset_dir, f"{mode}.parquet")
    print(f"Saving parquet to {out_path}")
    pq.write_table(table, out_path)


def prepare_neighbors(
    data_dir: str = "./dataset/random512d1m",
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
    print(f"Distances: {distances.shape}, Indices: {indices.shape}")

    assert all(indices[:, 0] == np.arange(len(test)))  ### first N vectors

    # save flat search result as neighbors
    df = pd.DataFrame({"id": np.arange(len(indices)), "neighbors_id": indices.tolist()})

    table = pa.Table.from_pandas(df)
    pq.write_table(table, f"{data_dir}/neighbors.parquet")
    print(f"Saving parquet to {data_dir}/neighbors.parquet")


if __name__ == "__main__":
    args = get_args()

    # generate random data and save as .npy
    vectors = get_random_data(
        num_data=args.dataset_size,
        dim=args.dim,
        seed=42,
    )

    # prepare train parquet file from numpy arrays
    npy_to_parquet(
        vector=vectors,
        dataset_dir=args.dataset_dir,
        mode="train",
    )

    # prepare test set
    test_vectors = vectors[: args.query_size]  ### first N vectors

    npy_to_parquet(
        vector=test_vectors,
        dataset_dir=args.dataset_dir,
        mode="test",
    )

    # prepare neighbors
    prepare_neighbors(data_dir=args.dataset_dir)
