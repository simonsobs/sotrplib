"""
Configuration for the streaming, per-map-preprocessed coadding tool
(sotrp-coadd). Deliberately separate from the main pipeline's Settings
(sotrplib.config.config): this pipeline shape has no forced photometry,
blind search, source catalogs, or sifter, so reusing Settings would just
mean a lot of unused fields.
"""

import logging
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .map_coadding import RhoKappaMapCoadderConfig
from .maps import MapCatDatabaseConfig
from .outputs import MapOutputConfig
from .preprocessors import AllPreprocessorConfigTypes


class CoaddRegistrationConfig(BaseModel):
    coadd_name: str
    "Human-readable, ideally unique name for this coadd."
    coadd_type: str = "depth1_streaming_coadd"
    enabled: bool = True
    "If False, build and save the coadd but skip writing to mapcat."


class CoaddSettings(BaseSettings):
    instrument: str = "LAT"

    maps: MapCatDatabaseConfig
    "Depth-1 maps to read and stream-coadd. map_type must be 'intensity' -- "
    "this tool does its own per-map matched filtering, so it needs raw "
    "intensity/inverse-variance maps as input, not already-filtered ones."

    preprocessors: list[AllPreprocessorConfigTypes] = []
    "Applied, in order, to each individual depth-1 map before it is merged "
    "into the running coadd (e.g. planet_mask, matched_filter, kappa_rho, "
    "edge_mask)."

    map_coadder: RhoKappaMapCoadderConfig = Field(
        default_factory=RhoKappaMapCoadderConfig
    )

    map_outputs: list[MapOutputConfig] = []

    mapcat_registration: CoaddRegistrationConfig | None = None
    "If set, register the finished coadd (and links to its constituent "
    "depth-1 maps) in mapcat's depth_one_coadds table."

    log_level: int | str = logging.INFO

    model_config = SettingsConfigDict(env_prefix="sotrp_coadd_", extra="ignore")

    @model_validator(mode="after")
    def _check_map_type(self) -> "CoaddSettings":
        if self.maps.map_type != "intensity":
            raise ValueError(
                "CoaddSettings.maps.map_type must be 'intensity' (raw "
                f"depth-1 maps to preprocess per-map), got '{self.maps.map_type}'. "
                "Already-filtered rho/kappa maps don't need this tool -- use "
                "sotrp's own preprocessors/map_coadder config instead."
            )
        return self

    @classmethod
    def from_file(cls, config_path: Path | str) -> "CoaddSettings":
        with open(config_path, "r") as handle:
            return cls.model_validate_json(handle.read())

    def to_dependencies(self) -> dict[str, Any]:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(self.log_level),
        )
        log = structlog.get_logger()

        return {
            "maps": self.maps.to_generator(log=log),
            "preprocessors": [x.to_preprocessor(log=log) for x in self.preprocessors],
            "coadder": self.map_coadder.to_coadder(log=log),
            "map_outputs": [x.to_output(log=log) for x in self.map_outputs],
        }
