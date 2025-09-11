"""
Dependencies for pre-processing (e.g. map cleaning). Affects
maps in-place. These operations happen in between 'build' and
'finalize' for the map creatino and i/o process.
"""

from abc import ABC, abstractmethod
from typing import Literal

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.maps import clean_map, kappa_clean


class MapPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        """
        Provides preprocessing for the input map
        """
        return input_map


class KappaRhoCleaner(MapPreprocessor):
    cut_on: Literal["median", "percentile", "max"] = "median"
    fraction: float = 0.05

    def __init__(
        self,
        cut_on: Literal["median", "percentile", "max"] = "median",
        fraction: float = 0.05,
    ):
        self.cut_on = cut_on
        self.fraction = fraction

    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        # TODO: Figure out what warnings this raises
        input_map.kappa = kappa_clean(kappa=input_map.kappa, rho=input_map.rho)
        input_map.rho = clean_map(
            imap=input_map.rho,
            inverse_variance=input_map.kappa,
            cut_on=self.cut_on,
            fraction=self.fraction,
        )

        return input_map
