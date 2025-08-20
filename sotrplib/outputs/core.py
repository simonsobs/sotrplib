"""
Dealing with sources once they've been found.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

from sotrplib.maps.core import ProcessableMap
from sotrplib.sifter.core import SifterResult
from sotrplib.sources.sources import SourceCandidate


class SourceOutput(ABC):
    @abstractmethod
    def output(
        self,
        forced_photometry_candidates: list[SourceCandidate],
        sifter_result: SifterResult,
        input_map: ProcessableMap,
    ):
        """
        Output the source candidates somehow. We also pass the
        input map in case e.g. we wish to reconstruct thumbnails from it.
        """
        return


class JSONSerializer(ABC):
    """
    Serialize a source candidate list to a JSON file.
    """

    filename: Path

    def __init__(self, filename: Path):
        self.filename = filename

    def output(
        self,
        forced_photometry_candidates: list[SourceCandidate],
        sifter_result: SifterResult,
        input_map: ProcessableMap,
    ):
        with self.filename.open("w") as handle:
            json.dump(
                obj={
                    "forced_photometry": forced_photometry_candidates,
                    "sifted_blind_serach": sifter_result,
                },
                fp=handle,
            )

        return
