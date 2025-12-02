import numpy as np
import pandas as pd
import argparse

# from sklearn.cluster import KMeans
from cuml.cluster import KMeans
from numpy.linalg import norm


def get_args():
    parser = argparse.ArgumentParser(description="KMeans Centroid Calculation")
    parser.add_argument(
        "--nlist",
        type=int,
        default=250,
        help="Number of clusters for KMeans",
    )
    return parser.parse_args()

def load_performance1536d_500k_dataset():
    file_path = "/tmp/vectordb_bench/dataset/openai/openai_medium_500k"
    
    print("Loading dataset from:", file_path)
    
    train_vectors = pd.read_parquet(f"{file_path}/shuffle_train.parquet")
    test_vectors = pd.read_parquet(f"{file_path}/test.parquet")
    neighbors = pd.read_parquet(f"{file_path}/neighbors.parquet")

    assert test_vectors.shape[0] == neighbors.shape[0], "Test and Neighbor sizes do not match"
    assert (test_vectors["id"] == neighbors["id"]).all(), "Test IDs do not match Neighbor IDs"
    
    train_vectors.sort_values(by="id", inplace=True)
    train_ids = train_vectors["id"].to_numpy(dtype=np.int64)
    train_vectors = np.vstack(train_vectors["emb"].values)
    print(f"train_vectors shape: {train_vectors.shape}")
    
    test_vectors = np.vstack(test_vectors['emb'].values)
    print(f"test_vectors shape: {test_vectors.shape}")
    
    neighbors = np.vstack(neighbors['neighbors_id'].values)
    print(f"neighbors shape: {neighbors.shape}")

    train_vectors /= norm(train_vectors, axis=1, keepdims=True)
    test_vectors /= norm(test_vectors, axis=1, keepdims=True)
    
    return train_vectors.astype(np.float32), test_vectors.astype(np.float32), neighbors.astype(np.int32).tolist(), train_ids


def main():
    args = get_args()
    
    nlist = args.nlist
    seed = 42

    # prepare dataset
    train_vectors, _, _, _ = load_performance1536d_500k_dataset()
    dim = train_vectors.shape[1]
    assert dim == 1536, "Dimension should be 1536"
    print("✅ Load dataset complete.")

    # kmeans
    kmeans = KMeans(n_clusters=nlist, random_state=seed)
    # kmeans = BalancedKMeans(n_clusters=nlist, random_state=seed)
    kmeans.fit(train_vectors)
    print("✅ KMeans training complete.")

    # allocate
    labels = kmeans.predict(train_vectors)
    print(f"Labels shape: {labels.shape}")
    num_vectors = []
    num_shards = 0
    for i in range(nlist):
        indices = np.where(labels == i)[0]
        print(f"Centroid {i}: {len(indices)} vectors assigned")
        num_vectors.append(len(indices))
        num_shards += (len(indices)-1) // 4096 + 1

    print(f"Max vectors in a cluster: {max(num_vectors)}")
    print(f"Min vectors in a cluster: {min(num_vectors)}")
    print(f"Total number of shards: {num_shards}")
    
    file_name = f"centroids_1536D500K_{nlist}.npy"
    centroids = kmeans.cluster_centers_
    centroids /= norm(centroids, axis=1, keepdims=True)
    print(f"Norm: {norm(centroids, axis=1)}")
    print(f"Centroids shape: {centroids.shape}")
    np.save(file_name, centroids)
    print(f"✅ Centroids saved to {file_name}")

if __name__ == "__main__":
    main()