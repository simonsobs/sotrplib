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
from pixell import enmap, utils
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

    @property
    def rho(self):
        """
        Calculate the rho map from snr and flux.
        """
        if getattr(self, "__rho", None) is not None:
            return self.__rho

        with np.errstate(divide="ignore"):
            rho = (self.snr * self.snr) / (self.flux)

        return rho

    @rho.setter
    def rho(self, x):
        # TODO: Set the snr, flux when this is updated?
        self.__rho = x

    @rho.deleter
    def rho(self, x):
        del self.__rho

    @property
    def kappa(self):
        """
        Calculate the kappa map from snr and flux.
        """
        if getattr(self, "__kappa", None) is not None:
            return self.__kappa

        with np.errstate(divide="ignore"):
            kappa = (self.snr * self.snr) / (self.flux * self.flux)

        return kappa

    @kappa.setter
    def kappa(self, x):
        # TODO: Set the snr, flux when this is updated?
        self.__kappa = x

    @kappa.deleter
    def kappa(self, x):
        del self.__kappa

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
            resolution=self.simulation_parameters.resolution.to_value("arcmin"),
            map_noise=self.simulation_parameters.map_noise.to_value("Jy"),
            log=log,
        )

        self.flux_units = u.Jy

        if (
            self.simulation_parameters.map_noise is not None
            and self.simulation_parameters.map_noise > 0.0
        ):
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
        box: ArrayLike | None = None,
        time_filename: Path | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.rho_filename = rho_filename
        self.kappa_filename = kappa_filename
        self.time_filename = time_filename
        self.start_time = start_time
        self.end_time = end_time
        self.box = box
        self.log = log or structlog.get_logger()

    def build(self):
        log = self.log.bind(rho_filename=self.rho_filename)
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


class CoaddedMap(ProcessableMap):
    """
    A set of FITS maps read from disk suitable for coadding.

    res in radian
    """

    def __init__(
        self,
        source_maps: list[ProcessableMap],
        log: FilteringBoundLogger | None = None,
    ):
        self.source_maps = source_maps
        self.log = log or structlog.get_logger()

        self.rho = None
        self.kappa = None
        self.input_map_times = []
        self.start_time = None
        self.end_time = None

    def build(self):
        for sourcemap in self.source_maps:
            sourcemap.build()
            if isinstance(self.rho, type(None)):
                self.rho = sourcemap.rho.copy()
                self.kappa = sourcemap.kappa.copy()
                self.time = sourcemap.time.copy() + sourcemap.start_time.timestamp()
                self.map_depth = sourcemap.time.copy()
                self.map_depth[self.map_depth > 0] = 1.0
                self.res = np.abs(sourcemap.rho.wcs.wcs.cdelt[0]) * utils.degree
                self.start_time = sourcemap.start_time.timestamp()
                self.end_time = sourcemap.end_time.timestamp()
                self.input_map_times.append(
                    0.5
                    * (
                        sourcemap.start_time.timestamp()
                        + sourcemap.end_time.timestamp()
                    )
                )
            else:
                self.rho = enmap.map_union(
                    self.rho,
                    sourcemap.rho,
                )
                self.kappa = enmap.map_union(
                    self.kappa,
                    sourcemap.kappa,
                )
                self.time = enmap.map_union(
                    self.time,
                    sourcemap.time + sourcemap.start_time.timestamp(),
                )
                self.start_time = min(self.start_time, sourcemap.start_time.timestamp())
                self.end_time = max(self.end_time, sourcemap.end_time.timestamp())
                self.input_map_times.append(
                    0.5
                    * (
                        sourcemap.start_time.timestamp()
                        + sourcemap.end_time.timestamp()
                    )
                )
                sourcemap_depth = sourcemap.time.copy()
                sourcemap_depth[sourcemap_depth > 0] = 1.0
                self.map_depth = enmap.map_union(
                    self.map_depth,
                    sourcemap_depth,
                )

        self.time /= self.map_depth
        self.n_maps = len(self.input_map_times)

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
