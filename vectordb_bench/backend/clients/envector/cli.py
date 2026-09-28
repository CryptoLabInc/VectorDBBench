from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

DBTYPE = DB.EnVector


class EnVectorTypedDict(TypedDict):
    """enVector common parameters"""

    uri: Annotated[
        str,
        click.option("--uri", type=str, help="uri connection string", required=True),
    ]
    eval_mode: Annotated[
        str,
        click.option(
            "--eval-mode",
            help="Evaluation mode",
            type=click.Choice(["mm", "mms", "mm32", "mms32", "rmp"]),
            default="mm32",
        ),
    ]
    preset: Annotated[
        str,
        click.option(
            "--preset",
            help="Parameter preset (must match --eval-mode: mm/mms->ip1, mm32/mms32->ip2 or ip3). "
            "Empty => derived from --eval-mode.",
            type=str,
            default="",
        ),
    ]
    index_name: Annotated[
        str,
        click.option("--index-name", help="Index name", type=str, default="vdbbench"),
    ]
    key_id: Annotated[
        str,
        click.option(
            "--key-id", help="enVector key id (KMS-managed or local keys/<id>/)", type=str, default="default_key"
        ),
    ]
    kms_address: Annotated[
        str,
        click.option(
            "--kms-address",
            type=str,
            default="",
            help="KMS gateway host:port; enables KMS-managed keys (empty = local keys)",
        ),
    ]
    kms_secure: Annotated[
        bool,
        click.option("--kms-secure", type=bool, default=False, help="Use TLS to the KMS gateway"),
    ]


class EnVectorFlatIndexTypedDict(CommonTypedDict, EnVectorTypedDict): ...


@cli.command(name="envectorflat")
@click_parameter_decorators_from_typed_dict(EnVectorFlatIndexTypedDict)
def EnVectorFlat(**parameters: Unpack[EnVectorFlatIndexTypedDict]):
    from .config import EnVectorConfig, FlatIndexConfig

    run(
        db=DBTYPE,
        db_config=EnVectorConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
            key_id=parameters["key_id"],
            eval_mode=parameters["eval_mode"],
            collection_name=parameters["index_name"],
            kms_address=parameters["kms_address"],
            kms_secure=parameters["kms_secure"],
            index_params={},
        ),
        db_case_config=FlatIndexConfig(eval_mode=parameters["eval_mode"], preset=parameters["preset"]),
        **parameters,
    )


class EnVectorIVFFlatIndexTypedDict(CommonTypedDict, EnVectorTypedDict):
    """IVF-FLAT index specific parameters"""

    nlist: Annotated[
        int,
        click.option("--nlist", type=int, help="nlist for IVF index", default=250),
    ]
    nprobe: Annotated[
        int,
        click.option("--nprobe", type=int, help="nprobe for IVF index", default=6),
    ]
    train_centroids: Annotated[
        bool,
        click.option("--train-centroids", type=bool, help="train IVF centroids", default=False),
    ]
    centroids_path: Annotated[
        str,
        click.option("--centroids-path", type=str, help="path to centroids for IVF index", default=None),
    ]


@cli.command(name="envectorivfflat")
@click_parameter_decorators_from_typed_dict(EnVectorIVFFlatIndexTypedDict)
def EnVectorIVFFlat(**parameters: Unpack[EnVectorIVFFlatIndexTypedDict]):
    from .config import EnVectorConfig, IVFFlatIndexConfig

    run(
        db=DBTYPE,
        db_config=EnVectorConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
            key_id=parameters["key_id"],
            eval_mode=parameters["eval_mode"],
            collection_name=parameters["index_name"],
            kms_address=parameters["kms_address"],
            kms_secure=parameters["kms_secure"],
            index_params={"nlist": parameters["nlist"], "nprobe": parameters["nprobe"]},
        ),
        db_case_config=IVFFlatIndexConfig(
            eval_mode=parameters["eval_mode"],
            preset=parameters["preset"],
            nlist=parameters["nlist"],
            nprobe=parameters["nprobe"],
            train_centroids=parameters["train_centroids"],
            centroids_path=parameters["centroids_path"],
        ),
        **parameters,
    )


class EnVectorIVFGASIndexTypedDict(CommonTypedDict, EnVectorIVFFlatIndexTypedDict): ...


@cli.command(name="envectorivfgas")
@click_parameter_decorators_from_typed_dict(EnVectorIVFGASIndexTypedDict)
def EnVectorIVFGAS(**parameters: Unpack[EnVectorIVFGASIndexTypedDict]):
    from .config import EnVectorConfig, IVFGASIndexConfig

    run(
        db=DBTYPE,
        db_config=EnVectorConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
            key_id=parameters["key_id"],
            eval_mode=parameters["eval_mode"],
            collection_name=parameters["index_name"],
            kms_address=parameters["kms_address"],
            kms_secure=parameters["kms_secure"],
            index_params={"nlist": parameters["nlist"], "nprobe": parameters["nprobe"]},
        ),
        db_case_config=IVFGASIndexConfig(
            eval_mode=parameters["eval_mode"],
            preset=parameters["preset"],
            nlist=parameters["nlist"],
            nprobe=parameters["nprobe"],
            train_centroids=parameters["train_centroids"],
            centroids_path=parameters["centroids_path"],
        ),
        **parameters,
    )
