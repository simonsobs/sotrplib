"""
Dealing with sources once they've been found.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

from sotrplib.sources.sources import SourceCandidate


class SourceOutput(ABC):
    @abstractmethod
    def output(self, candidates: list[SourceCandidate]):
        return


class JSONSerializer(ABC):
    filename: Path

    def __init__(self, filename: Path):
        self.filename = filename

    def output(self, candidates: list[SourceCandidate]):
        with self.filename.open("w") as handle:
            json.dump(obj=candidates, fp=handle)

        return
