"""
Dealing with sources once they've been found.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import SourceCandidate


class SourceOutput(ABC):
    @abstractmethod
    def output(self, candidates: list[SourceCandidate], input_map: ProcessableMap):
        """
        Output the source candidates somehow. We also pass the
        input map in case e.g. we wish to reconstruct thumbnails from it.
        """
        return


class JSONSerializer(ABC):
    filename: Path

    def __init__(self, filename: Path):
        self.filename = filename

    def output(self, candidates: list[SourceCandidate], input_map: ProcessableMap):
        with self.filename.open("w") as handle:
            json.dump(obj=candidates, fp=handle)

        return
