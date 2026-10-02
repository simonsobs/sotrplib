from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel, model_validator
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
    high_sig: float = 5.0
    "A confirmed event needs at least one detection with SNR >= high_sig."
    low_sig: float = 3.0
    "Candidates with SNR >= low_sig are matched; the other arrays can confirm "
    "an event with these detections. The blind search threshold and the "
    "sifter's snr cut must be <= low_sig to give these candidates."
    summary_directory: Path | None = None
    "If set, write each run's ranked event groups here as JSON."

    @model_validator(mode="after")
    def _check_thresholds(self):
        if self.low_sig > self.high_sig:
            raise ValueError(
                f"low_sig ({self.low_sig}) must be <= high_sig ({self.high_sig})"
            )
        return self

    def to_matcher(
        self, log: FilteringBoundLogger | None = None
    ) -> MultiArrayMapMatcher:
        return MultiArrayMapMatcher(
            radius=self.radius,
            min_arrays=self.min_arrays,
            high_sig=self.high_sig,
            low_sig=self.low_sig,
            summary_directory=self.summary_directory,
            log=log,
        )


AllMapMatcherConfigTypes = EmptyMapMatcherConfig | MultiArrayMapMatcherConfig
