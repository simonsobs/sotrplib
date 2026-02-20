from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from lightcurvedb.config import Settings as LightcurveDBSettings
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.outputs.core import (
    CutoutImageOutput,
    JSONSerializer,
    PickleSerializer,
    SourceOutput,
)
from sotrplib.outputs.lightcurvedb import LightcurveDBOutput
from sotrplib.outputs.lightserve import LightServeOutput


class OutputConfig(BaseModel, ABC):
    output_type: str

    @abstractmethod
    def to_output(self, log: FilteringBoundLogger | None = None) -> SourceOutput:
        return


class PickleOutputConfig(OutputConfig):
    output_type: Literal["pickle"] = "pickle"
    directory: Path

    def to_output(self, log: FilteringBoundLogger | None = None) -> PickleSerializer:
        return PickleSerializer(directory=self.directory)


class JSONOutputConfig(OutputConfig):
    output_type: Literal["json"] = "json"
    directory: Path

    def to_output(self, log: FilteringBoundLogger | None = None) -> JSONSerializer:
        return JSONSerializer(directory=self.directory)


class CutoutImageOutputConfig(OutputConfig):
    output_type: Literal["cutout"] = "cutout"
    directory: Path

    def to_output(self, log: FilteringBoundLogger | None = None) -> CutoutImageOutput:
        return CutoutImageOutput(directory=self.directory)


class LightServeOutputConfig(OutputConfig):
    output_type: Literal["lightserve"] = "lightserve"
    hostname: str
    token_tag: str | None = None
    identity_server: str | None = None

    def to_output(self, log: FilteringBoundLogger | None = None) -> SourceOutput:
        return LightServeOutput(
            hostname=self.hostname,
            token_tag=self.token_tag,
            identity_server=self.identity_server,
            log=log,
        )


class LightcurveDBOutputConfig(OutputConfig):
    output_type: Literal["lightcurvedb"] = "lightcurvedb"
    override_settings: dict | None = None
    "Over-ride individual settings from the LightcurveDB environment variable setup."
    upsert_sources: bool = False
    "Upsert sources to LightcurveDB before outputting flux measuurements. Data is taken from the crossmatches."

    def to_output(self, log: FilteringBoundLogger | None = None) -> SourceOutput:
        settings = LightcurveDBSettings(**(self.override_settings or {}))
        return LightcurveDBOutput(
            settings=settings, upsert_sources=self.upsert_sources, log=log
        )


AllOutputConfigTypes = (
    PickleOutputConfig
    | JSONOutputConfig
    | CutoutImageOutputConfig
    | LightServeOutputConfig
    | LightcurveDBOutputConfig
)
