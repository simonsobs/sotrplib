from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sifter.map_matching import (
    EmptyMapMatcher,
    MapMatcher,
    MultiArrayMapMatcher,
)


class MapMatcherConfig(BaseModel, ABC):
    matcher_type: str

    @abstractmethod
    def to_matcher(self, log: FilteringBoundLogger | None = None) -> MapMatcher:
        return


class EmptyMapMatcherConfig(MapMatcherConfig):
    matcher_type: Literal["empty"] = "empty"

    def to_matcher(self, log: FilteringBoundLogger | None = None) -> EmptyMapMatcher:
        return EmptyMapMatcher()


class MultiArrayMapMatcherConfig(MapMatcherConfig):
    matcher_type: Literal["multi_array"] = "multi_array"
    radius: AstroPydanticQuantity[u.arcmin] = 1.5 * u.arcmin
    "Maximum separation between detections of the same event in different maps."
    min_arrays: int = 2
    "Distinct arrays (optics tubes) an event must be detected in to stay a "
    "transient candidate."
    summary_directory: Path | None = None
    "If set, write each run's ranked event groups here as JSON."

    def to_matcher(
        self, log: FilteringBoundLogger | None = None
    ) -> MultiArrayMapMatcher:
        return MultiArrayMapMatcher(
            radius=self.radius,
            min_arrays=self.min_arrays,
            summary_directory=self.summary_directory,
            log=log,
        )


AllMapMatcherConfigTypes = EmptyMapMatcherConfig | MultiArrayMapMatcherConfig
