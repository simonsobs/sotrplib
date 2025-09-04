"""
Source simulations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pixell import enmap
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.maps import edge_map
from sotrplib.sims import sim_maps, sim_sources, sim_utils
from sotrplib.source_catalog.database import SourceCatalogDatabase
from sotrplib.sources.sources import RegisteredSource


class ProcessableMapWithSimulatedSources(ProcessableMap):
    """
    A ProcessableMap that has its sources simulated.
    """

    def __init__(
        self,
        flux: enmap.ndmap,
        snr: enmap.ndmap,
        time: enmap.ndmap,
        original_map: ProcessableMap,
    ):
        self.flux = flux
        self.snr = snr

        self.time_first = np.minimum(time, original_map.time_first)
        self.time_mean = time
        self.time_end = np.maximum(time, original_map.time_first)

        self.original_map = original_map

        self.observation_length = original_map.observation_length
        self.observation_start = original_map.observation_start
        self.observation_end = original_map.observation_end

        self.flux_units = original_map.flux_units
        self.frequency = original_map.frequency
        self.array = original_map.array
        self.finalized = original_map.finalized
        self.map_resolution = original_map.map_resolution

        return

    def build(self):
        return

    def finalize(self):
        super().finalize()
        return


class SourceSimulation(ABC):
    """
    Abstract base class for source simulations.
    """

    @abstractmethod
    def simulate(
        self, input_map: ProcessableMap
    ) -> tuple[ProcessableMap, list[RegisteredSource]]:
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
        simulated_sources
            The sources that were added.
        """
        return


class SourceSimulationPassthrough(SourceSimulation):
    """
    Don't actually simulate any sources!
    """

    def simulate(
        self, input_map: ProcessableMap
    ) -> tuple[ProcessableMap, list[RegisteredSource]]:
        return input_map, []


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

    def simulate(
        self, input_map: ProcessableMap
    ) -> tuple[ProcessableMap, list[RegisteredSource]]:
        noise_map = input_map.noise

        new_flux_map, injected_sources = sim_maps.inject_sources(
            imap=input_map.flux.copy(),
            sources=self.source_database.source_list,
            observation_time=input_map.time_mean,
            freq=input_map.frequency,
            arr=input_map.array,
            # TODO: Handle map IDs and debug (pass down logger)
            map_id=None,
            debug=True,
            log=self.log,
        )

        log = self.log.bind(n_sources=len(injected_sources))

        with np.errstate(divide="ignore"):
            snr = new_flux_map / noise_map

        log.info("source_injection.database.complete")

        return ProcessableMapWithSimulatedSources(
            flux=new_flux_map,
            snr=snr,
            time=input_map.time_mean,
            original_map=input_map,
        ), injected_sources


class RandomSourceSimulationParameters(BaseModel):
    n_sources: int
    "The number of sources to inject"
    min_flux: AstroPydanticQuantity[u.Jy]
    "Minimum flux for the sources"
    max_flux: AstroPydanticQuantity[u.Jy]
    "Maximum flux for the sources"
    fwhm_uncertainty_frac: float
    "Uncertainty in the FWHM"
    fraction_return: float
    "The fraction of sources to return (i.e. the fraction that are available to the forced photometry)"


class RandomSourceSimulation(SourceSimulation):
    """
    Inject a number of random sources.
    """

    log: FilteringBoundLogger
    parameters: RandomSourceSimulationParameters

    def __init__(
        self,
        parameters: RandomSourceSimulationParameters,
        log: FilteringBoundLogger | None = None,
    ):
        self.parameters = parameters
        self.log = log or structlog.get_logger()

    def simulate(
        self, input_map: ProcessableMap
    ) -> tuple[ProcessableMap, list[RegisteredSource]]:
        noise_map = input_map.noise
        log = self.log or structlog.get_logger()

        log = log.bind(parameters=self.parameters)

        # TODO: Store the source information as we should be injecting the
        # same sources in all maps in the run.
        new_flux_map, injected_sources = sim_maps.photutils_sim_n_sources(
            sim_map=input_map.flux.copy(),
            n_sources=self.parameters.n_sources,
            min_flux_Jy=self.parameters.min_flux.to_value("Jy"),
            max_flux_Jy=self.parameters.max_flux.to_value("Jy"),
            map_noise_Jy=0.0,  # Disabled now
            fwhm_uncert_frac=self.parameters.fwhm_uncertainty_frac,
            freq=input_map.frequency,
            arr=input_map.array,
            # TODO: Map IDs
            map_id=None,
            ctime=input_map.time_mean,
            return_registered_sources=True,
            log=log,
        )

        log = log.bind(n_sources=len(injected_sources))

        with np.errstate(divide="ignore"):
            snr = new_flux_map / noise_map

        log.info("source_injection.random.complete")

        return ProcessableMapWithSimulatedSources(
            flux=new_flux_map,
            snr=snr,
            time=input_map.time_mean,
            original_map=input_map,
        ), injected_sources[
            : int(len(injected_sources) * self.parameters.fraction_return)
        ]


class TransientDatabaseSourceSimulation(SourceSimulation):
    """
    Inject transient sources from a transient database.
    """

    log: FilteringBoundLogger
    transient_database_path: Path

    def __init__(
        self, transient_database_path: Path, log: FilteringBoundLogger | None = None
    ):
        self.transient_database_path = transient_database_path
        self.log = log or structlog.get_logger()

    def simulate(
        self, input_map: ProcessableMap
    ) -> tuple[ProcessableMap, list[RegisteredSource]]:
        noise_map = input_map.noise

        log = self.log.bind(transient_database_path=self.transient_database_path)

        transient_sources = sim_utils.load_transients_from_db(
            self.transient_database_path
        )

        log.bind(n_loaded=len(transient_sources))

        new_flux_map, injected_sources = sim_maps.inject_sources(
            imap=input_map.flux.copy(),
            sources=transient_sources,
            observation_time=input_map.time_mean,
            freq=input_map.frequency,
            arr=input_map.array,
            # TODO: map IDs
            map_id=None,
            debug=True,
            log=log,
        )

        log = log.bind(n_sources=len(injected_sources))

        with np.errstate(divide="ignore"):
            snr = new_flux_map / noise_map

        log.info("source_injection.transient_database.complete")

        return ProcessableMapWithSimulatedSources(
            flux=new_flux_map,
            snr=snr,
            time=input_map.time_mean,
            original_map=input_map,
        ), injected_sources


@dataclass
class TransientSourceSimulationParameters:
    n_transients: int
    min_flux: u.Quantity
    max_flux: u.Quantity
    min_width: u.Quantity = u.Quantity(0.1, "d")
    max_width: u.Quantity = u.Quantity(10.0, "d")


class TransientSourceSimulation(SourceSimulation):
    """
    Inject transient sources that are synthetically generated.
    """

    log: FilteringBoundLogger
    parameters: TransientSourceSimulationParameters

    def __init__(
        self,
        parameters: TransientSourceSimulationParameters,
        log: FilteringBoundLogger | None = None,
    ):
        self.parameters = parameters
        self.log = log or structlog.get_logger()

    def simulate(
        self, input_map: ProcessableMap
    ) -> tuple[ProcessableMap, list[RegisteredSource]]:
        noise_map = input_map.noise

        log = self.log.bind(parameters=self.parameters)

        hits_map = edge_map(np.nan_to_num(input_map.flux))

        transients_for_injection = sim_sources.generate_transients(
            n=self.parameters.n_transients,
            imap=hits_map,
            ra_lims=None,
            dec_lims=None,
            peak_amplitudes=(
                self.parameters.min_flux.to_value("Jy"),
                self.parameters.max_flux.to_value("Jy"),
            ),
            peak_times=(
                input_map.observation_start.timestamp(),
                input_map.observation_end.timestamp(),
            ),
            flare_widths=(
                self.parameters.min_width.to_value("d"),
                self.parameters.max_width.to_value("d"),
            ),
            uniform_on_sky=False,
        )

        log = log.bind(n_generated=len(transients_for_injection))

        new_flux_map, injected_sources = sim_maps.inject_sources(
            imap=input_map.flux.copy(),
            sources=transients_for_injection,
            observation_time=input_map.time_mean,
            freq=input_map.frequency,
            arr=input_map.array,
            # TODO: map_id
            map_id=None,
            debug=True,
            log=log,
        )
        log = log.bind(n_injected=len(injected_sources))

        with np.errstate(divide="ignore"):
            snr = new_flux_map / noise_map

        log.info("source_injection.transient_simulation.complete")

        return ProcessableMapWithSimulatedSources(
            flux=new_flux_map,
            snr=snr,
            time=input_map.time_mean,
            original_map=input_map,
        ), injected_sources
