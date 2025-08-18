"""
Source simulations
"""

from abc import ABC, abstractmethod

import numpy as np
import structlog
from pixell import enmap
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sims import sim_maps
from sotrplib.source_catalog.database import SourceCatalogDatabase
from sotrplib.sources.sources import SourceCandidate


class ProcessableMapWithSimualtedSources(ProcessableMap):
    """
    A ProcessableMap that has its sources simulated.
    """

    def __init__(
        self,
        flux: enmap.ndmap,
        snr: enmap.ndmap,
        time: enmap.ndmap | None,
        original_map: ProcessableMap,
    ):
        self.flux = flux
        self.snr = snr
        self.time = time

        self.original_map = original_map

        self.observation_length = original_map.observation_length
        self.start_time = original_map.start_time
        self.end_time = original_map.end_time
        self.flux_units = original_map.flux_units
        self.frequency = original_map.frequency

        return


class SourceSimulation(ABC):
    """
    Abstract base class for source simulations.
    """

    @abstractmethod
    def simulate(self, input_map: ProcessableMap) -> ProcessableMap:
        """
        Simulate sources on the input map.

        Parameters
        ----------
        input_map : ProcessableMap
            The map to simulate sources on.

        Returns
        -------
        ProcessableMap
            The map with simulated sources.
        """
        return


class SourceSimulationPassthrough(SourceSimulation):
    """
    Don't actually simulate any sources!
    """

    def simulate(
        self, input_map: ProcessableMap
    ) -> tuple[ProcessableMap, list[SourceCandidate]]:
        return input_map


class DatabaseSourceSimulation(SourceSimulation):
    """
    Inject a number of sources from a specific database.
    """

    log: FilteringBoundLogger
    source_database: SourceCatalogDatabase

    def __init__(
        self, source_database: SourceCatalogDatabase, log: FilteringBoundLogger
    ):
        self.source_database = source_database
        self.log = log or structlog.get_logger()

    def simulate(self, input_map: ProcessableMap) -> ProcessableMap:
        noise_map = input_map.noise

        new_flux_map, injected_sources = sim_maps.inject_sources(
            imap=input_map.flux.copy(),
            sources=self.source_database.source_list,
            observation_time=input_map.start_time.timestamp(),
            freq=input_map.frequency,
            arr=input_map.array,
            # TODO: Handle map IDs and debug (pass down logger)
            map_id=None,
            debug=True,
        )

        with np.errstate(divide="ignore"):
            snr = new_flux_map / noise_map

        return ProcessableMapWithSimualtedSources(
            flux=new_flux_map,
            snr=snr,
            time=input_map.time,
            original_map=input_map,
        ), injected_sources
