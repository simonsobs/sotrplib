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
    """Abstract base class for source-candidate output handlers."""

    @abstractmethod
    def output(
        self,
        forced_photometry_candidates: list[MeasuredSource],
        sifter_result: SifterResult,
        map_id: str,
        pointing_sources: list[MeasuredSource] = [],  # for compatibility
        injected_sources: list[SimulatedSource] = [],  # for compatibility
    ):
        """Write source candidates to an output destination.

        Parameters
        ----------
        forced_photometry_candidates : list of MeasuredSource
            Sources measured via forced photometry.
        sifter_result : SifterResult
            Sifted blind-search candidates.
        map_id : str
            Identifier for the input map.
        pointing_sources : list of MeasuredSource, optional
            Sources used for pointing calibration.
        injected_sources : list of SimulatedSource, optional
            Simulated sources injected into the map.
        """
        return


class MapOutput(ABC):
    """Abstract base class for map output handlers."""

    field_ids: list[str]

    @abstractmethod
    def output(self, input_map: ProcessableMap):
        """Write the processed map to an output destination.

        Parameters
        ----------
        input_map : ProcessableMap
            The processed map to output.
        """
        return


class JSONSerializer(SourceOutput):
    """Serialize source candidate lists to a timestamped JSON file.

    Parameters
    ----------
    directory : Path
        Directory where JSON files are written.
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
    """Serialize source candidate lists to a timestamped pickle file.

    Parameters
    ----------
    directory : Path
        Directory where pickle files are written.
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
    """Save cutout PNG images around each source candidate.

    Parameters
    ----------
    directory : Path
        Directory where cutout image files are written.
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


class MapOutputSerializer(MapOutput):
    """Serialize selected map fields to FITS files on disk.

    Parameters
    ----------
    directory : Path
        Directory where FITS files are written.
    field_ids : list of str
        Map attribute names (e.g. ``"flux"``, ``"snr"``) to serialize.
    log : FilteringBoundLogger, optional
        Structured logger.
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
