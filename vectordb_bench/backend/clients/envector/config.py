from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class EnVectorConfig(DBConfig):
    """enVector common configuration"""

    uri: SecretStr = SecretStr("http://localhost:50050")
    key_path: str = "keys"
    key_id: str = "default_key"
    collection_name: str = "vdbbench"

    def to_dict(self) -> dict:
        return {
            "uri": self.uri.get_secret_value(),
            "key_path": self.key_path,
            "key_id": self.key_id,
        }


class EnVectorIndexConfig(BaseModel):
    """Base index config for envector"""

    index: IndexType
    metric_type: MetricType = MetricType.COSINE  # envector supports cosine similarity only
    use_partition_key: bool = True  # for label-filter
    eval_mode: str = "mm"  # default eval_mode

    @property
    def is_gpu_index(self) -> bool:
        return self.index in [
            IndexType.GPU_CAGRA,
            IndexType.GPU_IVF_FLAT,
            IndexType.GPU_IVF_PQ,
            IndexType.GPU_BRUTE_FORCE,
        ]

    def parse_metric(self) -> str:
        if not self.metric_type:
            return ""

        if self.is_gpu_index and self.metric_type == MetricType.COSINE:
            return MetricType.L2.value
        return self.metric_type.value


class FlatIndexConfig(EnVectorIndexConfig, DBCaseConfig):
    """enVector FLAT index configuration"""

    index: IndexType = IndexType.Flat

    def index_param(self) -> dict:
        return {
            "metric_type": "COSINE",
            "index_type": self.index.value,
            "eval_mode": self.eval_mode,
            "params": {"index_type": "FLAT"},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": "COSINE",
            "search_params": {},
        }


class IVFFlatIndexConfig(EnVectorIndexConfig, DBCaseConfig):
    """enVector IVF-FLAT index configuration"""

    index: IndexType = IndexType.IVFFlat
    nlist: int = 0
    nprobe: int = 0
    train_centroids: bool = False  # whether to train centroids before inserting data
    centroids_path: str | None = None  # path to centroids file

    def index_param(self) -> dict:
        return {
            "metric_type": "COSINE",
            "index_type": self.index.value,
            "eval_mode": self.eval_mode,
            "params": {"index_type": "IVF_FLAT", "nlist": self.nlist, "default_nprobe": self.nprobe},
            "train_centroids": self.train_centroids,
            "centroids_path": self.centroids_path,
        }

    def search_param(self) -> dict:
        return {
            "metric_type": "COSINE",
            "search_params": {"nprobe": self.nprobe},
        }


class IVFGASIndexConfig(IVFFlatIndexConfig):
    """enVector IVF-GAS index configuration"""

    index: IndexType = IndexType.IVFGAS

    def index_param(self) -> dict:
        index_param = super().index_param()
        index_param["params"].update({"index_type": "IVF_VCT"})
        return index_param


_envector_case_config = {
    IndexType.Flat: FlatIndexConfig,
    IndexType.IVFFlat: IVFFlatIndexConfig,
    IndexType.IVFGAS: IVFGASIndexConfig,
}
