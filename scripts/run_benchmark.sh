#!/bin/bash

set -euo pipefail

export DATASET_LOCAL_DIR="./dataset"
export NUM_PER_BATCH=4096

CASE_TYPE="PerformanceCustomDataset"
DATASET_NAME="PUBMED768D400K"
CENTROID_PATH=centroids/embeddinggemma-300m/centroids.npy
VCT_PATH=centroids/embeddinggemma-300m/tree_info.pkl
ENVECTOR_URI="localhost:50050"
NLIST=32768
NPROBE=6

REQUESTED_TYPE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)
            REQUESTED_TYPE="${2:-}"
            shift 2
            ;;
        --type=*)
            REQUESTED_TYPE="${1#--type=}"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

case "$REQUESTED_TYPE" in
    ""|flat|ivf|ivf-trained|ivf-gas) ;;
    *)
        echo "Invalid --type: $REQUESTED_TYPE (expected: flat / ivf / ivf-trained / ivf-gas)" >&2
        exit 1
        ;;
esac
COMMON_ARGS=(
    --uri "$ENVECTOR_URI"
    --eval-mode mm
    --case-type "$CASE_TYPE"
    --custom-case-name "$DATASET_NAME"
    --custom-dataset-name "$DATASET_NAME"
    --custom-dataset-dir ""
    --custom-dataset-size 400335
    --custom-dataset-dim 768
    --custom-dataset-file-count 1
    --custom-dataset-with-gt
    --skip-custom-dataset-use-shuffled
    --k 10
)

run_case() {
    local engine=$1
    local label=$2
    shift 2
    python -m vectordb_bench.cli.vectordbbench "$engine" \
        "${COMMON_ARGS[@]}" \
        --db-label "$label" \
        "$@"
}

if [[ -z "$REQUESTED_TYPE" || "$REQUESTED_TYPE" == "flat" ]]; then
    run_case envectorflat "$DATASET_NAME-FLAT"
fi

if [[ -z "$REQUESTED_TYPE" || "$REQUESTED_TYPE" == "ivf" ]]; then
    export NUM_PER_BATCH=500000  # set database size for efficiency
    run_case envectorivfflat "$DATASET_NAME-IVF-RANDOM" \
        --nlist "$NLIST" \
        --nprobe "$NPROBE"
fi

if [[ -z "$REQUESTED_TYPE" || "$REQUESTED_TYPE" == "ivf-trained" ]]; then
    export NUM_PER_BATCH=500000  # set database size for efficiency
    run_case envectorivfflat "$DATASET_NAME-IVF-FLAT" \
        --train-centroids True \
        --centroids-path "$CENTROID_PATH" \
        --nlist "$NLIST" \
        --nprobe "$NPROBE"
fi

if [[ -z "$REQUESTED_TYPE" || "$REQUESTED_TYPE" == "ivf-gas" ]]; then
    export NUM_PER_BATCH=500000  # set database size for efficiency
    run_case envectorivfflat "$DATASET_NAME-IVF-GAS" \
        --is-vct True \
        --train-centroids True \
        --centroids-path "$CENTROID_PATH" \
        --vct-path "$VCT_PATH" \
        --nlist "$NLIST" \
        --nprobe "$NPROBE"
fi
