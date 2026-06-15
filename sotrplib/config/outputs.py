from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from lightcurvedb.config import Settings as LightcurveDBSettings
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.outputs.core import (
    CutoutImageOutput,
    JSONSerializer,
    MapOutputSerializer,
    PickleSerializer,
    SourceOutput,
)
from sotrplib.outputs.lightcurvedb import LightcurveDBOutput
from sotrplib.outputs.lightserve import LightServeOutput


class OutputConfig(BaseModel, ABC):
    """Abstract base for pipeline output configuration objects.

    Each subclass defines an ``output_type`` discriminator and implements
    ``to_output`` to construct a ``SourceOutput``.
    """

    output_type: str

    @abstractmethod
    def to_output(self, log: FilteringBoundLogger | None = None) -> SourceOutput:
        """Construct the configured output handler.

        Parameters
        ----------
        log : FilteringBoundLogger, optional
            Structured logger.

        Returns
        -------
        SourceOutput
            The constructed output handler instance.
        """
        return


class PickleOutputConfig(OutputConfig):
    """Configuration for pickle-file source output.

    Fields
    ------
    directory : Path
        Directory where pickle files are written.
    """

    output_type: Literal["pickle"] = "pickle"
    directory: Path

    def to_output(self, log: FilteringBoundLogger | None = None) -> PickleSerializer:
        return PickleSerializer(directory=self.directory)


class JSONOutputConfig(OutputConfig):
    """Configuration for JSON-file source output.

    Fields
    ------
    directory : Path
        Directory where JSON files are written.
    """

    output_type: Literal["json"] = "json"
    directory: Path

    def to_output(self, log: FilteringBoundLogger | None = None) -> JSONSerializer:
        return JSONSerializer(directory=self.directory)


class MapOutputConfig(OutputConfig):
    """Configuration for FITS map output serializer.

    Fields
    ------
    directory : Path
        Directory where map files are written.
    fields : list of str
        Map field IDs to serialize.
    """

    output_type: Literal["maps"] = "maps"
    directory: Path
    fields: list[str]

    def to_output(self, log: FilteringBoundLogger | None = None) -> MapOutputSerializer:
        return MapOutputSerializer(
            directory=self.directory, field_ids=self.fields, log=log
        )


class CutoutImageOutputConfig(OutputConfig):
    """Configuration for cutout image output.

    Fields
    ------
    directory : Path
        Directory where cutout image files are written.
    """

    output_type: Literal["cutout"] = "cutout"
    directory: Path

    def to_output(self, log: FilteringBoundLogger | None = None) -> CutoutImageOutput:
        return CutoutImageOutput(directory=self.directory)


class LightServeOutputConfig(OutputConfig):
    """Configuration for the LightServe HTTP output.

    Fields
    ------
    hostname : str
        LightServe server hostname.
    token_tag : str or None
        Tag used to look up the authentication token.
    identity_server : str or None
        Identity server URL for token exchange.
    """

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
    """Configuration for the LightcurveDB output handler.

    Fields
    ------
    override_settings : dict or None
        Key-value pairs that override individual LightcurveDB environment
        variable settings.
    upsert_sources : bool
        If ``True``, upsert sources to LightcurveDB before writing flux
        measurements (default ``False``).
    """

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
    | MapOutputConfig
    | CutoutImageOutputConfig
    | LightServeOutputConfig
    | LightcurveDBOutputConfig
)

SourceOutputConfigTypes = (
    PickleOutputConfig
    | JSONOutputConfig
    | LightcurveDBOutputConfig
    | LightServeOutputConfig
)

MapOutputConfigTypes = MapOutputConfig | CutoutImageOutputConfig
