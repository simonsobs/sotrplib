"""
Dealing with sources once they've been found.
"""

import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from sotrplib.maps.core import ProcessableMap
from sotrplib.sifter.core import SifterResult
from sotrplib.sources.sources import MeasuredSource


class SourceOutput(ABC):
    @abstractmethod
    def output(
        self,
        forced_photometry_candidates: list[MeasuredSource],
        sifter_result: SifterResult,
        input_map: ProcessableMap,
    ):
        """
        Output the source candidates somehow. We also pass the
        input map in case e.g. we wish to reconstruct thumbnails from it.
        """
        return


class JSONSerializer(SourceOutput):
    """
    Serialize a source candidate list to a JSON file.
    """

    directory: Path

    def __init__(self, directory: Path):
        self.directory = directory

    def output(
        self,
        forced_photometry_candidates: list[MeasuredSource],
        sifter_result: SifterResult,
        input_map: ProcessableMap,
    ):
        filename = (
            self.directory
            / f"{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}.json"
        )
        with filename.open("w") as handle:
            json.dump(
                obj={
                    "forced_photometry": forced_photometry_candidates,
                    "sifted_blind_search": sifter_result,
                },
                fp=handle,
            )

        return


class PickleSerializer(SourceOutput):
    """
    Serialize a source candidate list to a Pickle file.
    """

    directory: Path

    def __init__(self, directory: Path):
        self.directory = directory

    def output(
        self,
        forced_photometry_candidates: list[MeasuredSource],
        sifter_result: SifterResult,
        input_map: ProcessableMap,
    ):
        filename = (
            self.directory
            / f"{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}.pickle"
        )
        with filename.open("wb") as handle:
            pickle.dump(
                obj={
                    "forced_photometry": forced_photometry_candidates,
                    "sifted_blind_search": sifter_result,
                },
                file=handle,
            )

        return
