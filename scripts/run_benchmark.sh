#!/bin/bash

set -euo pipefail

DEFAULT_INDEX_TYPE="FLAT"
DEFAULT_CONFIG_FILE="envector_random_config.yml"

# requested parameters
REQUESTED_INDEX_TYPE="${INDEX_TYPE:-$DEFAULT_INDEX_TYPE}"    # default mode FLAT
REQUESTED_CONFIG_FILE="${CONFIG_FILE:-$DEFAULT_CONFIG_FILE}" # default config file

usage() {
  cat >&2 <<EOF
Usage: $0 [--config-file <path>] [--index-type <FLAT|IVF_FLAT|IVF_GAS>]

Benchmark enVector with specified index type and configuration file.

Options:
  --config-file  Path to the benchmark configuration file.
  --index-type   Index type to benchmark:
                   FLAT     : brute-force search (baseline).
                   IVF_FLAT : Inverted file (IVF) index with FLAT vectors for faster approximate search.
                   IVF_GAS  : enVector-customized ANN algorithm for the fastest approximate search.

Example:
  $0 --config-file envector_random_config.yml --index-type FLAT
  $0 --config-file envector_openai_config.yml --index-type IVF_FLAT
  $0 --config-file envector_pubmed_config.yml --index-type IVF_GAS
EOF
}

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file)
      REQUESTED_CONFIG_FILE="${2:-}"
      shift 2
      ;;
    --index-type)
      REQUESTED_INDEX_TYPE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [ -z "${REQUESTED_CONFIG_FILE}" ]; then
  echo "--config-file requires a path" >&2
  exit 1
fi

if [ -z "${REQUESTED_INDEX_TYPE}" ]; then
  echo "--index-type requires a value (FLAT / IVF_FLAT / IVF_GAS)" >&2
  exit 1
fi

# run benchmark based on requested type
case "${REQUESTED_INDEX_TYPE}" in
  FLAT)
    python -m vectordb_bench.cli.vectordbbench envectorflat \
      --config-file "${REQUESTED_CONFIG_FILE}"
    ;;
  IVF_FLAT)
    python -m vectordb_bench.cli.vectordbbench envectorivfflat \
      --config-file "${REQUESTED_CONFIG_FILE}"
    ;;
  IVF_GAS)
    python -m vectordb_bench.cli.vectordbbench envectorivfgas \
      --config-file "${REQUESTED_CONFIG_FILE}"
    ;;
  *)
    echo "Unsupported INDEX_TYPE: ${REQUESTED_INDEX_TYPE} (use FLAT or IVF_FLAT or IVF_GAS)" >&2
    exit 1
    ;;
esac
