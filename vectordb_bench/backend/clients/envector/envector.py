"""Wrapper around the EnVector vector database over VectorDB"""

import logging
import os
import time
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pyenvector as ev

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import EnVectorIndexConfig

log = logging.getLogger(__name__)

DROP_WAIT_TIMEOUT = float(os.environ.get("ENVECTOR_DROP_WAIT_TIMEOUT", "120"))
DROP_WAIT_POLL_INTERVAL = float(os.environ.get("ENVECTOR_DROP_WAIT_POLL_INTERVAL", "2"))

# Time to wait for the post-insert merge to reach MERGED_SAVED during optimize().
# Default 1 day: large index merges can run for many minutes/hours, and search
# must not start until the merge has fully cut over (see _optimize).
OPTIMIZE_WAIT_TIMEOUT = float(os.environ.get("ENVECTOR_OPTIMIZE_WAIT_TIMEOUT", "86400"))
OPTIMIZE_WAIT_POLL_INTERVAL = float(os.environ.get("ENVECTOR_OPTIMIZE_WAIT_POLL_INTERVAL", "2"))


class EnVector(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: EnVectorIndexConfig,
        collection_name: str = "vdbbench",
        drop_old: bool = False,
        name: str = "EnVector",
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the envector vector database."""
        self.name = name
        self.db_config = db_config
        self.case_config = db_case_config
        # Get collection_name from db_config if available, otherwise use the parameter
        self.collection_name = db_config.get("collection_name", collection_name)

        self._primary_field = "pk"
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self._vector_field = "vector"
        self._vector_index_name = "vector_idx"
        self._scalar_id_index_name = "id_sort_idx"
        self._scalar_labels_index_name = "labels_idx"
        self.col: ev.Index | None = None
        self._request_ids = []

        # Initialize the EnVector client
        ev.init(**self._client_init_kwargs())

        # Drop old index if specified
        if drop_old:
            log.info(f"{self.name} client drop_old index: {self.collection_name}")
            if self.collection_name in ev.get_index_list():
                ev.drop_index(self.collection_name)
                self._wait_until_index_deleted()

        # Check index type
        index_param = self.case_config.index_param().get("params", {})
        index_type = index_param.get("index_type", "FLAT")
        log.debug(f"Index Type: {index_type}")

        # Ensure the index exists or create it
        index_kwargs = dict(kwargs)
        self._ensure_index(dim, index_kwargs)

        ev.disconnect()

    def _client_init_kwargs(self) -> dict[str, Any]:
        """Common ev.init() kwargs. When kms_address is configured, the client
        routes key setup/decryption through the enVector KMS gateway instead of
        local key files; otherwise the original local-key behavior is kept."""
        kwargs: dict[str, Any] = {
            "address": self.db_config.get("uri"),
            "key_id": self.db_config.get("key_id"),
            "eval_mode": self.case_config.eval_mode,
        }
        # preset is optional: when unset pyenvector derives the per-eval_mode
        # default (mm/mms->ip1, mm32/mms32->ip2). Pass it only when the config
        # specifies one, so callers can override (e.g. "ip3" for mm32/mms32).
        if self.case_config.preset:
            kwargs["preset"] = self.case_config.preset
        kms_address = self.db_config.get("kms_address")
        if kms_address:
            # KMS manages the key material via auto_key_setup. Do NOT pass
            # key_path, or the client builds the cipher from local key files
            # (keys/<key_id>/EncKey.json) instead of the KMS-provided keys.
            kwargs["kms_address"] = kms_address
            # pyenvector uses a dedicated kms_secure flag (default True). Pass it
            # explicitly so a no-TLS KMS gateway isn't dialed over TLS.
            kwargs["kms_secure"] = bool(self.db_config.get("kms_secure", False))
        else:
            kwargs["key_path"] = self.db_config.get("key_path")
        return kwargs

    def _ensure_index(self, dim: int, index_kwargs: dict[str, Any]):
        # Check if the collection already exists
        if self.collection_name in ev.get_index_list():
            log.info(f"{self.name} index {self.collection_name} already exists, skip creating")
            return
        # Create the index if it does not exist
        self._create_index(dim, index_kwargs)

    def _create_index(self, dim: int, index_kwargs: dict[str, Any]):
        # Create the collection
        log.info(f"{self.name} create index: {self.collection_name}")

        index_param = self.case_config.index_param().get("params", {})
        index_type = index_param.get("index_type", "FLAT")
        train_centroids = self.case_config.index_param().get("train_centroids", False)

        if index_type in ["IVF_FLAT", "IVF_VCT"] and train_centroids:
            self._configure_centroids(index_param, index_kwargs)

        create_kwargs: dict[str, Any] = {
            "index_name": self.collection_name,
            "dim": dim,
            "key_id": self.db_config.get("key_id"),
            "index_params": index_param,
            "eval_mode": self.case_config.eval_mode,
        }
        # preset optional: unset => pyenvector derives the per-eval_mode default.
        if self.case_config.preset:
            create_kwargs["preset"] = self.case_config.preset
        # In KMS mode the key material comes from the gateway; passing key_path
        # would force local key-file lookup. Keep it only for local-key runs.
        if not self.db_config.get("kms_address"):
            create_kwargs["key_path"] = self.db_config.get("key_path")
        ev.create_index(**create_kwargs, **index_kwargs)

    def _wait_until_index_deleted(self):
        deadline = time.monotonic() + DROP_WAIT_TIMEOUT
        while time.monotonic() < deadline:
            if self.collection_name not in ev.get_index_list():
                log.info(f"{self.name} index {self.collection_name} deletion completed")
                return
            log.debug(f"{self.name} index {self.collection_name} still deleting; waiting...")
            time.sleep(DROP_WAIT_POLL_INTERVAL)

        msg = f"Timed out waiting for index deletion: {self.collection_name} (timeout={DROP_WAIT_TIMEOUT}s)"
        raise TimeoutError(msg)

    def _configure_centroids(self, index_param: dict[str, Any], index_kwargs: dict[str, Any]):
        # Load centroids
        centroid_path = self.case_config.index_param().get("centroids_path", None)
        if centroid_path is None:
            raise ValueError("Centroids path must be provided for IVF index training.")

        centroid_file = Path(centroid_path)
        if not centroid_file.exists():
            msg = f"Centroid file {centroid_path} not found for IVF index training."
            raise FileNotFoundError(msg)

        centroids = np.load(centroid_file)
        log.info(f"{self.name} loaded centroids from {centroid_path} for IVF index training.")

        index_param["centroids"] = centroids.tolist()

    @contextmanager
    def init(self):
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        ev.init(**self._client_init_kwargs())
        try:
            self.col = ev.Index(self.collection_name)
            yield
        finally:
            self.col = None
            ev.disconnect()

    def create_index(self):
        pass

    def _optimize(self):
        log.debug("Triggering indexing")
        self.col.indexing()
        log.info("Waiting for merge completion (target_stage=segmentation -> MERGED_SAVED)")
        self.col.wait_for_insert_stage(
            request_ids=self._request_ids,
            target_stage="segmentation",
            timeout_s=OPTIMIZE_WAIT_TIMEOUT,
            poll_interval_s=OPTIMIZE_WAIT_POLL_INTERVAL,
        )
        # clear request_ids after waiting
        self._request_ids = []
        log.info("Load index")
        self.col.load()
        log.info("enVector Indexing completed")

    def _post_insert(self):
        pass

    def optimize(self, data_size: int | None = None):
        assert self.col, "Please call self.init() before"
        self._optimize()

    def need_normalize_cosine(self) -> bool:
        """Whether this database need to normalize dataset to support COSINE"""
        return True

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into EnVector. should call self.init() first"""
        # use the first insert_embeddings to init collection
        assert self.col is not None
        assert len(embeddings) == len(metadata)

        # out-list filled with server-generated request_ids; _optimize waits on them
        request_ids = kwargs.pop("request_ids", [])

        insert_count = 0
        try:
            metadata = list(map(str, metadata))
            log.debug(f"Inserting {len(embeddings)} embeddings...")
            self.col.insert(
                embeddings, metadata, request_ids=request_ids, await_completion=False, execute_until="flush", load=False
            )
            insert_count += len(embeddings)
            log.debug(f"Insert successful, count={insert_count}")
        except Exception as e:
            log.exception("Failed to insert data")
            return insert_count, e
        return insert_count, None

    def set_request_ids(self, request_ids: list[str]):
        self._request_ids = request_ids

    def prepare_filter(self, filters: Filter):
        pass

    def search_embedding(
        self,
        query: list[float],
        k: int = 10,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return results."""
        assert self.col is not None

        try:
            # Perform the search.
            res = self.col.search(
                query=query,
                top_k=k,
                output_fields=["metadata"],
                search_params=self.case_config.search_param().get("search_params", {}),
            )

            # Handle empty results
            if not res or len(res) == 0:
                log.warning(f"Empty search results for query with k={k}")
                return []

            # Extract metadata from results
            # res structure: [[{id: X, score: Y, metadata: Z}, ...]]
            log.debug(f"Search results: {res[0][:1]}")  # Log first 1 results for debugging
            if not (res and len(res[0]) > 0):
                log.warning(f"Unexpected result structure: {res}")
                return []
            return [int(result["metadata"]) for result in res[0] if "metadata" in result]

        except Exception:
            log.exception("Search failed")
            return []
