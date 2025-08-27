from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from sotrplib.handlers.basic import PipelineRunner

from .blind_search import AllBlindSearchConfigTypes, EmptyBlindSearchConfig
from .forced_photometry import AllForcedPhotometryConfigTypes, EmptyPhotometryConfig
from .maps import AllMapConfigTypes
from .outputs import AllOutputConfigTypes
from .postprocessors import AllPostprocessorConfigTypes
from .preprocessors import AllPreprocessorConfigTypes
from .sifter import AllSifterConfigTypes, EmptySifterConfig
from .source_subtractor import (
    AllSourceSubtractorConfigTypes,
    EmptySourceSubtractorConfig,
)


class Settings(BaseSettings):
    # Telescope choice
    observatory: Literal["SO", "ACT", "SPT"] = "SO"
    instrument: Literal["LAT", "ACTPol", "SPT3G", "SAT1", "SAT2", "SAT3"] = "LAT"

    maps: list[AllMapConfigTypes] = []
    "Input maps"

    preprocessors: list[AllPreprocessorConfigTypes] = []
    "Map pre-processors"

    postprocessors: list[AllPostprocessorConfigTypes] = []
    "Map post-processors"

    forced_photometry: AllForcedPhotometryConfigTypes = Field(
        default_factory=EmptyPhotometryConfig
    )
    "Forced photometry settings"

    source_subtractor: AllSourceSubtractorConfigTypes = Field(
        default_factory=EmptySourceSubtractorConfig
    )
    "Source subtraction settings"

    blind_search: AllBlindSearchConfigTypes = Field(
        default_factory=EmptyBlindSearchConfig
    )
    "Blind search settings"

    sifter: AllSifterConfigTypes = Field(default_factory=EmptySifterConfig)
    "Sifting settings"

    outputs: list[AllOutputConfigTypes] = []
    "Source output settings"

    # Read environment and command line settings to override default
    model_config = SettingsConfigDict(env_prefix="sotrp_", extra="ignore")

    # Read json config settings to override default + environment
    @classmethod
    def from_file(cls, config_path: Path | str) -> "Settings":
        with open(config_path, "r") as handle:
            return cls.model_validate_json(handle.read())

    def to_dependencies(self) -> dict[str, Any]:
        log = structlog.get_logger()

        return {
            "maps": [x.to_map(log=log) for x in self.maps],
            "preprocessors": [x.to_preprocessor(log=log) for x in self.preprocessors],
            "postprocessors": [
                x.to_postprocessor(log=log) for x in self.postprocessors
            ],
            "forced_photometry": self.forced_photometry.to_forced_photometry(log=log),
            "source_subtractor": self.source_subtractor.to_source_subtractor(log=log),
            "blind_search": self.blind_search.to_search_provider(log=log),
            "sifter": self.sifter.to_sifter(log=log),
            "outputs": [x.to_output(log=log) for x in self.outputs],
        }

    def to_basic(self) -> PipelineRunner:
        return PipelineRunner(**self.to_dependencies())
