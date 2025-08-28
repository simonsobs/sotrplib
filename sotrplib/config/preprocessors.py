from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.maps.preprocessor import KappaRhoCleaner, MapPreprocessor


class PreprocessorConfig(BaseModel, ABC):
    preprocessor_type: str

    @abstractmethod
    def to_preprocessor(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPreprocessor:
        return


class KappaRhoCleanerConfig(PreprocessorConfig):
    preprocessor_type: Literal["kappa_rho"] = "kappa_rho"
    cut_on: Literal["median", "percentile", "max"] = "median"
    fraction: float = 0.05

    def to_preprocessor(
        self, log: FilteringBoundLogger | None = None
    ) -> KappaRhoCleaner:
        return KappaRhoCleaner(cut_on=self.cut_on, fraction=self.cut_on)


AllPreprocessorConfigTypes = KappaRhoCleanerConfig
