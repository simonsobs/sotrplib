from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from sotrplib.config.source_catalog import (
    AllSourceCatalogConfigTypes,
)
from sotrplib.handlers.basic import PipelineRunner
from sotrplib.handlers.prefect import PrefectRunner

from .blind_search import AllBlindSearchConfigTypes, EmptyBlindSearchConfig
from .forced_photometry import AllForcedPhotometryConfigTypes, EmptyPhotometryConfig
from .maps import AllMapGeneratorConfigTypes
from .outputs import AllOutputConfigTypes
from .postprocessors import AllPostprocessorConfigTypes
from .preprocessors import AllPreprocessorConfigTypes
from .sifter import AllSifterConfigTypes, EmptySifterConfig
from .source_injector import AllSourceInjectorConfigTypes, EmptySourceInjectorConfig
from .source_simulation import AllSourceSimulationConfigTypes
from .source_subtractor import (
    AllSourceSubtractorConfigTypes,
    EmptySourceSubtractorConfig,
)


class Settings(BaseSettings):
    # Telescope choice
    observatory: Literal["SO", "ACT", "SPT"] = "SO"
    instrument: Literal["LAT", "ACTPol", "SPT3G", "SAT1", "SAT2", "SAT3"] = "LAT"

    maps: AllMapGeneratorConfigTypes = []
    "Input maps"

    source_simulators: list[AllSourceSimulationConfigTypes] = []
    "Any source simulators to add sources to the map"

    source_injector: AllSourceInjectorConfigTypes = Field(
        default_factory=EmptySourceInjectorConfig
    )
    "Source injector to use for adding simulated sources to maps"

    source_catalogs: list[AllSourceCatalogConfigTypes] = []
    "Source catalogs to use"

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

        contents = {
            "maps": [x.to_map(log=log) for x in self.maps],
            "source_simulators": [
                x.to_simulator(log=log) for x in self.source_simulators
            ],
            "source_injector": self.source_injector.to_injector(log=log),
            "source_catalogs": [
                x.to_source_catalog(log=log) for x in self.source_catalogs
            ],
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
        return contents

    def to_basic(self) -> PipelineRunner:
        return PipelineRunner(**self.to_dependencies())

    def to_prefect(self) -> PrefectRunner:
        return PrefectRunner(**self.to_dependencies())
