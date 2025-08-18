"""
Core map objects.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import astropy.units as u
import numpy as np
import structlog
from astropy.units import Quantity, Unit
from numpy.typing import ArrayLike
from pixell import enmap
from pixell.enmap import ndmap
from structlog.types import FilteringBoundLogger

from sotrplib.sims import sim_maps


class ProcessableMap(ABC):
    snr: ndmap
    "The signal-to-noise ratio map"
    flux: ndmap
    "The flux map"
    time: ndmap | None
    "The time map. If None, usually when the map is a coadd, see the time range"

    frequency: str
    "The frequency band of the map, e.g. f090"
    array: str
    "The array/wafer that was used"

    observation_length: timedelta
    start_time: datetime
    end_time: datetime

    flux_units: Unit

    @abstractmethod
    def build(self):
        """
        Build the maps for snr, flux, and time. Could include simulating them,
        or reading them from disk.
        """
        return

    @property
    def noise(self):
        """
        Get the 'noise map' (flux / SNR)
        """
        with np.errstate(divide="ignore"):
            return self.flux / self.snr

    @abstractmethod
    def finalize(self):
        """
        Called after source injection to ensure that the snr, flux, and time maps
        are available.
        """
        return


@dataclass
class SimulationParameters:
    """
    Sky simulation parameters (geometry and noise level).
    Defaults are given to be the center of the Deep56 ACT field.
    """

    center_ra: u.Quantity = u.Quantity(16.3, "deg")
    center_dec: u.Quantity = u.Quantity(-1.8, "deg")
    width_ra: u.Quantity = u.Quantity(2.0, "deg")
    width_dec: u.Quantity = u.Quantity(1.0, "deg")
    resolution: u.Quantity = u.Quantity(0.5, "arcmin")
    map_noise: u.Quantity = u.Quantity(0.01, "Jy")


class SimulatedMap(ProcessableMap):
    def __init__(
        self,
        resolution: Quantity,
        start_time: datetime,
        end_time: datetime,
        frequency: str | None = None,
        array: str | None = None,
        simulation_parameters: SimulationParameters | None = None,
        box: ArrayLike | None = None,
        include_half_pixel_offset: bool = False,
        log: FilteringBoundLogger | None = None,
    ):
        """
        Parameters
        ----------
        resolution: Quantity
            The angular resolution of the map.
        start_time: datetime
            Start time of the simulated observing session.
        end_time: datetime
            End time of the simulated observing session.
        frequency: str | None, optional
            The frequency band of the simulation, e.g. f090. Defaults to f090.
        array: str | None, optional
            The array that this map represents.
        simulation_parameters: SimulationParameters, optional
            Parameters for the simulation. If None, defaults will be used.
        box: np.ndarray, optional
            Optional sky box to simulate in. Otherwise covers the whole sky.
        include_half_pixel_offset: bool, False
            Include the half-pixel offset in pixell constructors.
        log: FilteringBoundLogger, optional
            Logger to use. If None, a new one will be created.
        """
        self.resolution = resolution
        self.start_time = start_time
        self.end_time = end_time
        self.box = box
        self.include_half_pixel_offset = include_half_pixel_offset
        self.frequency = frequency or "f090"
        self.array = array or "pa5"

        self.observation_length = end_time - start_time
        self.simulation_parameters = simulation_parameters or SimulationParameters()
        self.time = None
        self.log = log or structlog.get_logger()

    def build(self):
        log = self.log.bind(parameters=self.simulation_parameters)

        self.flux = sim_maps.make_enmap(
            center_ra=self.simulation_parameters.center_ra.to_value("deg"),
            center_dec=self.simulation_parameters.center_dec.to_value("deg"),
            width_ra=self.simulation_parameters.width_ra.to_value("deg"),
            width_dec=self.simulation_parameters.width_dec.to_value("deg"),
            resolution=self.resolution.to_value("arcmin"),
            map_noise=self.simulation_parameters.map_noise.to_value("Jy"),
        )

        self.flux_units = u.Jy

        if self.simulation_parameters.map_noise:
            log.debug(
                "simulated_map.build.noise",
                map_noise=self.simulation_parameters.map_noise.to_value("Jy"),
            )
            self.snr = self.flux / self.simulation_parameters.map_noise.to_value("Jy")
        else:
            log.debug("simulated_map.build.noise.none")
            self.snr = self.flux.copy()
            self.snr.fill(1.0)  # Set SNR to 1 (i.e. all noise).

        log.debug("simulated_map.build.complete")

        return

    def finalize(self):
        # Nothing to do here.
        return


class SimulatedMapFromGeometry(ProcessableMap):
    def __init__(
        self,
        resolution: Quantity,
        start_time: datetime,
        end_time: datetime,
        geometry_source_map: Path,
        frequency: str | None = None,
        array: str | None = None,
        map_noise: Quantity = u.Quantity(0.01, "Jy"),
        log: FilteringBoundLogger | None = None,
    ):
        """
        Parameters
        ----------
        resolution: Quantity
            The angular resolution of the map.
        start_time: datetime
            Start time of the simulated observing session.
        end_time: datetime
            End time of the simulated observing session.
        geometry_source_map: Path
            Path to the source map that defines the geometry of the simulation.
        frequency: str | None, optional
            The frequency band of the simulation, e.g. f090. Defaults to f090.
        array: str | None, optional
            The array that this represents, defaults to pa5
        map_noise: Quantity, optional
            The noise level of the simulated map. Defaults to 0 Jy.
        log: FilteringBoundLogger, optional
            Logger to use. If None, a new one will be created.
        """
        self.resolution = resolution
        self.start_time = start_time
        self.end_time = end_time

        self.map_noise = map_noise
        self.geometry_source_map = geometry_source_map
        self.frequency = frequency or "f090"
        self.array = array or "pa5"

        self.observation_length = end_time - start_time
        self.time = None
        self.log = log or structlog.get_logger()

    def build(self):
        log = self.log.bind(geometry_source_map=self.geometry_source_map)

        # Read the geometry source map to get the center and width.
        # TODO: Potentially optimize by actually just reading the header.
        geom_map = enmap.read_map(str(self.geometry_source_map))

        log.debug("simulated_map.build.geometry.read")

        self.flux = sim_maps.make_noise_map(
            imap=geom_map,
            map_noise_jy=self.map_noise.to_value("Jy"),
        )

        self.flux_units = u.Jy

        if self.map_noise:
            log.debug(
                "simulated_map.build.noise",
                map_noise=self.map_noise.to_value("Jy"),
            )
            self.snr = self.flux / self.map_noise.to_value("Jy")
        else:
            log.debug("simulated_map.build.noise.none")
            self.snr = self.flux.copy()
            self.snr.fill(1.0)

        log.debug("simulated_map.build.complete")

        return

    def finalize(self):
        # Nothing to do here.
        return


class IntensityAndInverseVarianceMap(ProcessableMap):
    """
    A set of FITS maps read from disk. Could be Depth 1, could
    be monthly or weekly co-adds. Or something else!
    """

    def __init__(
        self,
        intensity_filename: Path,
        inverse_variance_filename: Path,
        start_time: datetime,
        end_time: datetime,
        box: ArrayLike | None,
        time_filename: Path | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.intensity_filename = intensity_filename
        self.inverse_variance_filename = inverse_variance_filename
        self.time_filename = time_filename
        self.start_time = start_time
        self.end_time = end_time
        self.box = box
        self.log = log or structlog.get_logger()

    def build(self):
        log = self.log.bind(intensity_filename=self.intensity_filename)
        try:
            self.intensity = enmap.read_map(
                str(self.intensity_filename), sel=0, box=self.box
            )
            log.debug("intensity_ivar.intensity.read.sel")
        except IndexError:
            # Intensity map does not have Q, U
            self.intensity = enmap.read_map(str(self.intensity_filename), box=self.box)
            log.debug("intensity_ivar.intensity.read.nosel")

        # TODO: Set metadata from header e.g. frequency band.

        log = log.new(inverse_variance_filename=self.inverse_variance_filename)
        self.inverse_variance = enmap.read_map(
            self.inverse_variance_filename, box=self.box
        )
        log.debug("intensity_viar.ivar.read")

        log = log.new(time_filename=self.time_filename)
        if self.time_filename is not None:
            # TODO: Handle nuance that the start time is not included.
            self.time = enmap.read_map(self.time_filename, box=self.box)
            log.debug("intensity_ivar.time.read")
        else:
            self.time = None
            log.debug("intensity_ivar.time.none")

        return

    def finalize(self):
        # Do matched filtering to get snr and flux maps.
        return


class RhoAndKappaMap(ProcessableMap):
    """
    A set of FITS maps read from disk. Could be Depth 1, could
    be monthly or weekly co-adds. Or something else!
    """

    def __init__(
        self,
        rho_filename: Path,
        kappa_filename: Path,
        start_time: datetime,
        end_time: datetime,
        box: ArrayLike | None,
        time_filename: Path | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.rho_filename = rho_filename
        self.inverse_variance_filename = kappa_filename
        self.time_filename = time_filename
        self.start_time = start_time
        self.end_time = end_time
        self.box = box
        self.log = log or structlog.get_logger()

    def build(self):
        log = self.log.bind(intensity_filename=self.intensity_filename)
        try:
            self.rho = enmap.read_map(str(self.rho_filename), sel=0, box=self.box)
            log.debug("rho_kappa.rho.read.sel")
        except IndexError:
            # Rho map does not have Q, U
            self.rho = enmap.read_map(str(self.rho_filename), box=self.box)
            log.debug("rho_kappa.rho.read.nosel")

        log = log.new(kappa_filename=self.kappa_filename)
        self.kappa = enmap.read_map(self.kappa_filename, box=self.box)
        log.debug("rho_kappa.kappa.read")

        # TODO: Set metadata from header e.g. frequency band.

        log = log.new(time_filename=self.time_filename)
        if self.time_filename is not None:
            # TODO: Handle nuance that the start time is not included.
            self.time = enmap.read_map(self.time_filename, box=self.box)
            log.debug("rho_kappa.time.read")
        else:
            self.time = None
            log.debug("rho_kappa.time.none")

        return

    def get_snr(self):
        with np.errstate(divide="ignore"):
            snr = self.rho / np.sqrt(self.kappa)

        return snr

    def get_flux(self):
        with np.errstate(divide="ignore"):
            flux = self.rho / self.kappa

        return flux

    def finalize(self):
        self.snr = self.get_snr()
        self.flux = self.get_flux()
