import argparse
import os

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import wget


def get_args():
    parser = argparse.ArgumentParser(description="Prepare LAION dataset for benchmarking.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.join(os.environ.get("DATASET_LOCAL_DIR", "/tmp/vectordb_bench/dataset"), "laion512d500k"),
        help="Directory to save the LAION dataset embeddings.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=500_000,
        help="Number of dataset embeddings to use.",
    )
    parser.add_argument(
        "--query-size",
        type=int,
        default=1_000,
        help="Number of query embeddings to use. 1,000 is recommended in VectorDBBench.",
    )

    return parser.parse_args()


def download_dataset(
    number: int = 1,
    download_dir: str = "./dataset/LAION512D500K",
) -> None:
    """Download LAION dataset embeddings from LAION deployment."""
    dataset_base_url = "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings"

    # file urls
    os.makedirs(download_dir, exist_ok=True)
    img_emb_path = f"{dataset_base_url}/img_emb/img_emb_{number}.npy"
    txt_emb_path = f"{dataset_base_url}/text_emb/text_emb_{number}.npy"
    metadata_path = f"{dataset_base_url}/metadata/metadata_{number}.parquet"

    # download
    if not os.path.exists(img_emb_path):
        wget.download(img_emb_path, out=download_dir)

    if not os.path.exists(txt_emb_path):
        wget.download(txt_emb_path, out=download_dir)

    if not os.path.exists(metadata_path):
        wget.download(metadata_path, out=download_dir)


def combine_npy_files(
    numbers: list[int] = [1],
    modal: str = "img",
    dataset_dir: str = "./dataset/LAION512D500K",
    dataset_size: int = 500_000,
) -> str:
    """Combine multiple .npy embedding files into a single .npy file."""
    if modal not in ["img", "text"]:
        raise ValueError("Modal should be either 'img' or 'text'.")

    if len(numbers) == 1:
        return f"{modal}_emb_{numbers[0]}.npy"

    all_arrays = []
    for i in numbers:
        arr = np.load(os.path.join(dataset_dir, f"{modal}_emb_{i}.npy"))
        all_arrays.append(arr)

    combined_arr = np.vstack(all_arrays)[:dataset_size]
    np.save(os.path.join(dataset_dir, f"{modal}_emb_combined_{combined_arr.shape[0]}.npy"), combined_arr)

    return f"{modal}_emb_combined_{combined_arr.shape[0]}.npy"


def npy_to_parquet(
    dataset_file: str = "img_emb_1.npy",
    modal: str = "img",
    dataset_dir: str = "./dataset/LAION512D500K",
    dataset_size: int = 500_000,
    query_size: int = 1_000,
) -> None:
    """Convert downloaded .npy embeddings to Parquet format."""
    print(f"Preparing {modal} embeddings from {dataset_file}...")

    if modal not in ["img", "text"]:
        raise ValueError("Modal should be either 'img' or 'text'.")

    if modal == "img":
        out_file = "train.parquet"
    elif modal == "text":
        out_file = "test.parquet"

    arr = np.load(f"{dataset_dir}/{dataset_file}")
    print(f"\tShape: {arr.shape}", end=" -> ")
    arr = arr[:dataset_size]
    print(f"{arr.shape}")

    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / norm
    print(f"\tNorm: {np.linalg.norm(arr[0])}")

    if modal == "text":
        np.random.seed(42)
        test_indices = np.random.randint(0, dataset_size, size=query_size)
        arr = arr[test_indices]

        test_out_path = os.path.join(dataset_dir, "test_indices.npy")
        np.save(test_out_path, test_indices)
        print(f"Saved test indices to {test_out_path}")

    ids = np.arange(len(arr))
    id_array = pa.array(ids, type=pa.int64())

    list_arrays = [arr[i].tolist() for i in range(len(arr))]
    vector_array = pa.array(list_arrays, type=pa.list_(pa.float64()))

    assert len(id_array) == len(vector_array)

    table = pa.Table.from_arrays([id_array, vector_array], names=["id", "emb"])

    out_path = os.path.join(dataset_dir, out_file)
    print(f"Saving parquet to {out_path}")
    pq.write_table(table, out_path)


def prepare_neighbors(
    data_dir: str = "./dataset/LAION512D500K",
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


if __name__ == "__main__":
    args = get_args()

    num_dataset_idx = (args.dataset_size - 1) // 1_000_000 + 1
    print(f"Downloading {num_dataset_idx} dataset files...")

    # download dataset files
    data_idx = [i for i in range(1, num_dataset_idx + 1)]
    for n in data_idx:
        download_dataset(number=n, download_dir=args.dataset_dir)
    print("\nDownload completed.")

    # combine npy files
    dataset_file_img = combine_npy_files(
        numbers=data_idx,
        modal="img",
        dataset_dir=args.dataset_dir,
        dataset_size=args.dataset_size,
    )
    dataset_file_text = combine_npy_files(
        numbers=data_idx,
        modal="text",
        dataset_dir=args.dataset_dir,
        dataset_size=args.dataset_size,
    )

    # combine npy files
    npy_to_parquet(
        dataset_file=dataset_file_img,
        modal="img",
        dataset_dir=args.dataset_dir,
        dataset_size=args.dataset_size,
        query_size=args.query_size,
    )
    npy_to_parquet(
        dataset_file=dataset_file_text,
        modal="text",
        dataset_dir=args.dataset_dir,
        dataset_size=args.query_size,
        query_size=args.query_size,
    )

    # prepare neighbors
    prepare_neighbors(data_dir=args.dataset_dir)
