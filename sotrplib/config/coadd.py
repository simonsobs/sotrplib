"""
Configuration for sotrp-coadd. This tool does not use the pipeline Settings,
because it has no forced photometry, blind search, catalogs or sifter.
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
    "A unique name for the coadd."
    coadd_type: str = "depth1_streaming_coadd"
    enabled: bool = True
    "If False, write the coadd to FITS but do not register it in mapcat."


class CoaddMapCatDatabaseConfig(MapCatDatabaseConfig):
    track_processing: bool = False
    "If True, skip completed maps and write a status for each map."


class CoaddSettings(BaseSettings):
    instrument: str = "LAT"

    maps: CoaddMapCatDatabaseConfig
    "The raw depth-1 maps for the coadd (map_type 'intensity')."

    preprocessors: list[AllPreprocessorConfigTypes] = []
    "The preprocessors to apply to each depth-1 map before the merge."

    map_coadder: RhoKappaMapCoadderConfig = Field(
        default_factory=RhoKappaMapCoadderConfig
    )

    map_outputs: list[MapOutputConfig] = []

    mapcat_registration: CoaddRegistrationConfig | None = None
    "If set, register the coadd and its depth-1 map links in mapcat."

    log_level: int | str = logging.INFO

    model_config = SettingsConfigDict(env_prefix="sotrp_coadd_", extra="ignore")

    @model_validator(mode="after")
    def _check_map_type(self) -> "CoaddSettings":
        if self.maps.map_type != "intensity":
            raise ValueError(
                "CoaddSettings.maps.map_type must be 'intensity', not "
                f"'{self.maps.map_type}'. For rho/kappa maps, use the "
                "map_coadder config of sotrp."
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
