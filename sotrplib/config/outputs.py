from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.outputs.core import JSONSerializer, PickleSerializer, SourceOutput


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


AllOutputConfigTypes = PickleOutputConfig
