"""
Dealing with sources once they've been found.
"""

import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from sotrplib.sifter.core import SifterResult
from sotrplib.sims.sim_sources import SimulatedSource
from sotrplib.sources.sources import MeasuredSource


class SourceOutput(ABC):
    @abstractmethod
    def output(
        self,
        forced_photometry_candidates: list[MeasuredSource],
        sifter_result: SifterResult,
        map_id: str,
        pointing_sources: list[MeasuredSource] = [],  # for compatibility
        injected_sources: list[SimulatedSource] = [],  # for compatibility
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
        pointing_sources: list[MeasuredSource] = [],  # for compatibility
        injected_sources: list[SimulatedSource] = [],  # for compatibility
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
                    "pointing_sources": pointing_sources,
                    "injected_sources": injected_sources,
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
        map_id: str,
        pointing_sources: list[MeasuredSource] = [],  # for compatibility
        injected_sources: list[SimulatedSource] = [],  # for compatibility
    ):
        filename = (
            self.directory
            / f"{map_id}_{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}.pickle"
        )
        with filename.open("wb") as handle:
            pickle.dump(
                obj={
                    "map_id": map_id,
                    "forced_photometry": forced_photometry_candidates,
                    "sifted_blind_search": sifter_result,
                    "pointing_sources": pointing_sources,
                    "injected_sources": injected_sources,
                },
                file=handle,
            )

        return


class CutoutImageOutput(SourceOutput):
    """
    Output cutout images around each source candidate.
    """

    directory: Path

    def __init__(self, directory: Path):
        self.directory = directory

    def output(
        self,
        forced_photometry_candidates: list[MeasuredSource],
        sifter_result: SifterResult,
        map_id: str,
        pointing_sources: list[MeasuredSource] = [],  # for compatibility
        injected_sources: list[SimulatedSource] = [],  # for compatibility
    ):
        for source in forced_photometry_candidates:
            cutout = source.thumbnail

            if cutout is None:
                continue

            filename = (
                self.directory / f"forced_photometry_{map_id}_{source.source_id}.png"
            )
            plt.imsave(fname=filename, arr=cutout, cmap="viridis")

        for ii, source in enumerate(
            sifter_result.source_candidates + sifter_result.transient_candidates
        ):
            cutout = source.thumbnail

            if cutout is None:
                continue

            if source.crossmatches:
                id = source.crossmatches[0].source_id
                filename = self.directory / f"blind_search_matched_{map_id}_{id}.png"
            else:
                filename = self.directory / f"blind_search_{map_id}_{ii}.png"

            plt.imsave(fname=filename, arr=cutout, cmap="viridis")

        return
