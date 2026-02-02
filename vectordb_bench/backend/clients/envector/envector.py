"""Wrapper around the EnVector vector database over VectorDB"""

import logging
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
        self.collection_name = collection_name

        self._primary_field = "pk"
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self._vector_field = "vector"
        self._vector_index_name = "vector_idx"
        self._scalar_id_index_name = "id_sort_idx"
        self._scalar_labels_index_name = "labels_idx"
        self.col: ev.Index | None = None

        # Initialize the EnVector client
        ev.init(
            address=self.db_config.get("uri"),
            key_path=self.db_config.get("key_path"),
            key_id=self.db_config.get("key_id"),
            eval_mode=self.case_config.eval_mode,
            preset="ip1" if self.case_config.eval_mode == "mm" else "ip",
        )

        # Drop old index if specified
        if drop_old:
            log.info(f"{self.name} client drop_old index: {self.collection_name}")
            if self.collection_name in ev.get_index_list():
                ev.drop_index(self.collection_name)

        # Check index type
        index_param = self.case_config.index_param().get("params", {})
        index_type = index_param.get("index_type", "FLAT")
        log.debug(f"Index Type: {index_type}")

        # Ensure the index exists or create it
        index_kwargs = dict(kwargs)
        self._ensure_index(dim, index_kwargs)

        ev.disconnect()

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

        ev.create_index(
            index_name=self.collection_name,
            dim=dim,
            key_path=self.db_config.get("key_path"),
            key_id=self.db_config.get("key_id"),
            index_params=index_param,
            eval_mode=self.case_config.eval_mode,
            **index_kwargs,
        )

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
        ev.init(
            address=self.db_config.get("uri"),
            key_path=self.db_config.get("key_path"),
            key_id=self.db_config.get("key_id"),
            eval_mode=self.case_config.eval_mode,
            preset="ip1" if self.case_config.eval_mode == "mm" else "ip",
        )
        try:
            self.col = ev.Index(self.collection_name)
            yield
        finally:
            self.col = None
            ev.disconnect()

    def create_index(self):
        pass

    def _optimize(self):
        pass

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

        request_ids = kwargs.pop("request_ids", [])  # extract request_ids from kwargs for tracking insert operations

        insert_count = 0
        try:
            metadata = list(map(str, metadata))
            log.debug(f"Inserting {len(embeddings)} embeddings...")
            self.col.insert(embeddings, metadata, request_ids=request_ids, await_completion=False)
            insert_count += len(embeddings)
            log.debug(f"Insert successful, count={insert_count}")
        except Exception as e:
            log.exception("Failed to insert data")
            return insert_count, e
        return insert_count, None

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
