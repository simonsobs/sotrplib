"""
Dealing with sources once they've been found.
"""

import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from pixell import enmap
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sifter.core import SifterResult
from sotrplib.sims.sim_sources import SimulatedSource
from sotrplib.sources.sources import MeasuredSource


class SourceOutput(ABC):
    @abstractmethod
    def output(
        self,
        forced_photometry_candidates: list[MeasuredSource],
        sifter_result: SifterResult,
        input_map: ProcessableMap,
        pointing_sources: list[MeasuredSource] = [],  # for compatibility
        injected_sources: list[SimulatedSource] = [],  # for compatibility
    ):
        """
        Output the source candidates somehow. We also pass the
        input map in case e.g. we wish to reconstruct thumbnails from it.
        """
        return


class MapOutput(ABC):
    field_ids: list[str]

    @abstractmethod
    def output(self, input_map: ProcessableMap):
        """
        Output the map after postprocessing. This is for outputs that want to
        do something with the map itself, e.g. save it to disk.
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
        input_map: ProcessableMap,
        pointing_sources: list[MeasuredSource] = [],  # for compatibility
        injected_sources: list[SimulatedSource] = [],  # for compatibility
    ):
        filename = (
            self.directory
            / f"{input_map.get_map_str_id()}_{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}.pickle"
        )
        with filename.open("wb") as handle:
            pickle.dump(
                obj={
                    "map_id": input_map.get_map_str_id(),
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
        input_map: ProcessableMap,
        pointing_sources: list[MeasuredSource] = [],  # for compatibility
        injected_sources: list[SimulatedSource] = [],  # for compatibility
    ):
        for source in forced_photometry_candidates:
            cutout = source.thumbnail

            if cutout is None:
                continue

            filename = self.directory / f"forced_photometry_{source.source_id}.png"
            plt.imsave(fname=filename, arr=cutout, cmap="viridis")

        for ii, source in enumerate(
            sifter_result.source_candidates + sifter_result.transient_candidates
        ):
            cutout = source.thumbnail

            if cutout is None:
                continue

            if source.crossmatches:
                id = source.crossmatches[0].source_id
                filename = self.directory / f"blind_search_matched_{id}.png"
            else:
                filename = self.directory / f"blind_search_{ii}.png"

            plt.imsave(fname=filename, arr=cutout, cmap="viridis")

        return


class MapOutputSerializer(MapOutput):
    """
    Output the map after postprocessing. This is for outputs that want to
    do something with the map itself, e.g. save it to disk.
    """

    directory: Path
    field_ids: list[str]

    def __init__(
        self,
        directory: Path,
        field_ids: list[str],
        log: FilteringBoundLogger | None = None,
    ):
        self.directory = directory
        self.field_ids = field_ids
        self.log = log or get_logger()

    def output(self, input_map: ProcessableMap):
        log = self.log
        # make sure output directory exists
        self.directory.mkdir(parents=True, exist_ok=True)
        for field_id in self.field_ids:
            if hasattr(input_map, field_id):
                map_to_save = getattr(input_map, field_id)
                if map_to_save is None:
                    log.error(
                        "MapOutputSerializer.field_is_none",
                        field_id=field_id,
                        map_id=input_map.get_map_str_id(),
                    )
                    continue
                filename = (
                    self.directory / f"{input_map.get_map_str_id()}_{field_id}.fits"
                )
                enmap.write_map(str(filename), map_to_save)
                log.info(
                    "MapOutputSerializer.saved_map",
                    field_id=field_id,
                    map_id=input_map.get_map_str_id(),
                    filename=str(filename),
                )
            else:
                log.error(
                    "MapOutputSerializer.failed_to_load",
                    field_id=field_id,
                    map_id=input_map.get_map_str_id(),
                )
