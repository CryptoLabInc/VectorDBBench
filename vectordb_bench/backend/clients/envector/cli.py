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
        click.option("--eval-mode", help="Evaluation mode", type=click.Choice(["mm", "rmp"]), default="mm"),
    ]
    index_name: Annotated[
        str,
        click.option("--index-name", help="Index name", type=str, default="vdbbench"),
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
            eval_mode=parameters["eval_mode"],
            collection_name=parameters["index_name"],
            index_params={},
        ),
        db_case_config=FlatIndexConfig(),
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
            eval_mode=parameters["eval_mode"],
            collection_name=parameters["index_name"],
            index_params={"nlist": parameters["nlist"], "nprobe": parameters["nprobe"]},
        ),
        db_case_config=IVFFlatIndexConfig(
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
            eval_mode=parameters["eval_mode"],
            collection_name=parameters["index_name"],
            index_params={"nlist": parameters["nlist"], "nprobe": parameters["nprobe"]},
        ),
        db_case_config=IVFGASIndexConfig(
            nlist=parameters["nlist"],
            nprobe=parameters["nprobe"],
            train_centroids=parameters["train_centroids"],
            centroids_path=parameters["centroids_path"],
        ),
        **parameters,
    )
