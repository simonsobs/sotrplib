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
    """
    Maps that are processable by the pipeline. They have two key properties:

    a) snr - The signal-to-noise map
    b) flux - The flux map

    Maps are created in two steps. First, 'build' reads the map, and then
    'finalize' converts its internals to signal-to-noise ratios and flux maps.

    We do this because you may want to read in maps of many different kinds,
    e.g. intensity and inverse variance, or rho and kappa maps, before converting
    them to the type that the pipeline needs.

    The values `rho`:`kappa` are settable. Keep in mind though,
    once you've called finalize, these maps will be deleted from memory and are
    no longer _settable_. You cannot update these maps because they would be
    inconsistent with the flux map that was generated.

    Note that `intensity`:`ivar` maps are not yet supported.
    """

    snr: ndmap
    "The signal-to-noise ratio map"
    flux: ndmap
    "The flux map"
    time_first: ndmap
    "The time at which each pixel was first observed"
    time_end: ndmap
    "The time at which each pixel was last observed"
    time_mean: ndmap
    "The mean time at which each pixel was observed"
    hits: ndmap
    "A hits map stating the number of times each pixel was observed"

    finalized: bool = False
    "Whether finalize has been called and ancillary maps can no longer be updated"

    frequency: str
    "The frequency band of the map, e.g. f090"
    array: str
    "The array/wafer that was used"

    observation_length: timedelta
    "Total length of the observation"
    observation_start: datetime
    "Start time of the observation"
    observation_end: datetime
    "End time of the observation"
    observation_time: datetime
    "Rough 'middle' time of the observation"

    flux_units: Unit

    map_resolution: u.Quantity | None

    __rho: enmap.ndmap | None = None
    __kappa: enmap.ndmap | None = None

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
        if self.finalized:
            raise AttributeError(
                "Secondary attributes are no longer accessible once you've finalized the map",
                name="rho",
            )

        if self.__rho is not None:
            return self.__rho

        with np.errstate(divide="ignore"):
            rho = (self.snr * self.snr) / (self.flux)

        return rho

    @rho.setter
    def rho(self, x):
        if self.finalized:
            raise AttributeError(
                "Secondary attributes are no longer accessible once you've finalized the map",
                name="kappa",
            )

        self.__rho = x

    @rho.deleter
    def rho(self):
        try:
            del self.__rho
        except AttributeError:
            pass

    @property
    def kappa(self):
        """
        Calculate the kappa map from snr and flux.
        """
        if self.finalized:
            raise AttributeError(
                "Secondary attributes are no longer accessible once you've finalized the map",
                name="kappa",
            )

        if self.__kappa is not None:
            return self.__kappa

        with np.errstate(divide="ignore"):
            kappa = (self.snr * self.snr) / (self.flux * self.flux)

        return kappa

    @kappa.setter
    def kappa(self, x):
        self.__kappa = x

    @kappa.deleter
    def kappa(self):
        try:
            del self.__kappa
        except AttributeError:
            pass

    @property
    def res(self):
        if self.map_resolution is not None:
            return self.map_resolution
        else:
            for attribute in ["flux", "snr", "rho", "kappa"]:
                if x := getattr(self, attribute, None):
                    res = abs(x.wcs.wcs.cdelt[0])
                    break

        self.map_resolution = res * u.degree

        return self.map_resolution

    @abstractmethod
    def finalize(self):
        """
        Called after source injection to ensure that the snr, flux, and time maps
        are available.
        """
        del self.rho
        del self.kappa

        self.finalized = True

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
        observation_start: datetime,
        observation_end: datetime,
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
        observation_start: datetime
            Start time of the simulated observing session.
        observation_end: datetime
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
        self.observation_start = observation_start
        self.observation_end = observation_end
        self.box = box
        self.include_half_pixel_offset = include_half_pixel_offset
        self.frequency = frequency or "f090"
        self.array = array or "pa5"

        self.observation_length = observation_end - observation_start
        self.observation_time = observation_start + 0.5 * self.observation_length
        self.simulation_parameters = simulation_parameters or SimulationParameters()
        self.log = log or structlog.get_logger()

    def build(self):
        log = self.log.bind(parameters=self.simulation_parameters)

        # Flux
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

        self.map_resolution = self.simulation_parameters.resolution

        # SNR
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

        # Time simulation is simple: a unique timestamp for each
        # pixel with time increasing mainly across RA.
        time_map = sim_maps.make_time_map(
            imap=self.flux,
            start_time=self.observation_start,
            end_time=self.observation_end,
        )
        self.time_first = time_map
        self.time_end = time_map
        self.time_mean = time_map

        log.debug("simulated_map.build.time")

        # Hits
        self.hits = (np.abs(self.flux) > 0.0).astype(int)
        log.debug("simulated_map.build.hits")

        log.debug("simulated_map.build.complete")

        return

    def finalize(self):
        super().finalize()


class SimulatedMapFromGeometry(ProcessableMap):
    def __init__(
        self,
        resolution: Quantity,
        geometry_source_map: Path,
        start_time: datetime | None,
        end_time: datetime | None,
        time_map_filename: Path | None = None,
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
        geometry_source_map: Path
            Path to the source map that defines the geometry of the simulation.
        time_map_filename: Path | None, optional
            A time map to use, if not present simulated from start time and end time.
        start_time: datetime | None, optional
            Start time of the simulated observing session.
        end_time: datetime | None, optional
            End time of the simulated observing session.
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
        self.observation_start = start_time
        self.observation_end = end_time

        self.map_noise = map_noise
        self.geometry_source_map = geometry_source_map
        self.time_map_filename = time_map_filename
        self.frequency = frequency or "f090"
        self.array = array or "pa5"

        if time_map_filename is None and start_time is None and end_time is None:
            raise RuntimeError(
                "One of time_map_filename or start_time and end_time must be provided"
            )

        self.observation_length = end_time - start_time
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

        self.map_resolution = self.resolution

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

        if self.time_map_filename is not None:
            time_map = enmap.read_map(str(self.time_map_filename))
        else:
            time_map = sim_maps.make_time_map(
                imap=self.flux,
                start_time=self.observation_start,
                end_time=self.observation_end,
            )

        self.time_first = time_map
        self.time_end = time_map
        self.time_mean = time_map

        log.debug("simulated_map.build.time")

        # Hits
        self.hits = (np.abs(self.flux) > 0.0).astype(int)
        log.debug("simulated_map.build.hits")

        log.debug("simulated_map.build.complete")

        return

    def finalize(self):
        super().finalize()


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
        self.observation_start = start_time
        self.observation_end = end_time
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
        log.debug("intensity_ivar.ivar.read")
        self.map_resolution = self.res()
        log = log.new(time_filename=self.time_filename)
        if self.time_filename is not None:
            # TODO: Handle nuance that the start time is not included.
            time_map = enmap.read_map(self.time_filename, box=self.box)
            log.debug("intensity_ivar.time.read")
        else:
            time_map = None
            log.debug("intensity_ivar.time.none")

        self.time_first = time_map
        self.time_end = time_map
        self.time_mean = time_map

        return

    def finalize(self):
        super().finalize()


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
        frequency: str | None = None,
        array: str | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.rho_filename = rho_filename
        self.kappa_filename = kappa_filename
        self.time_filename = time_filename
        self.observation_start = start_time
        self.observation_end = end_time
        self.box = box
        self.frequency = frequency
        self.array = array
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
        self.map_resolution = self.res()

        log = log.new(time_filename=self.time_filename)
        if self.time_filename is not None:
            # TODO: Handle nuance that the start time is not included.
            time_map = enmap.read_map(self.time_filename, box=self.box)
            log.debug("rho_kappa.time.read")
        else:
            time_map = None
            log.debug("rho_kappa.time.none")

        self.time_first = time_map
        self.time_end = time_map
        self.time_mean = time_map

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
        super().finalize()


"""
Need to update this to use individual time maps and calcaulte the time_start, time_mean and time_end maps.
"""


class CoaddedMap(ProcessableMap):
    """
    A set of FITS maps read from disk suitable for coadding.

    res in radian
    """

    def __init__(
        self,
        source_maps: list[ProcessableMap],
        log: FilteringBoundLogger | None = None,
        time_start: enmap.ndmap | None = None,
        time_mean: enmap.ndmap | None = None,
        time_end: enmap.ndmap | None = None,
        map_depth: enmap.ndmap | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        input_map_times: list[float] | None = None,
        frequency: str | None = None,
        array: str | None = None,
    ):
        self.source_maps = source_maps
        self.log = log or structlog.get_logger()
        self.initialized = False

        self.time_start = time_start
        self.time_mean = time_mean
        self.time_end = time_end
        self.map_depth = map_depth
        self.observation_start = start_time
        self.observation_end = end_time
        self.input_map_times = input_map_times or []
        self.frequency = frequency
        self.array = array

    def build(self):
        base_map = self.source_maps[0]
        base_map.build()
        base_map.finalize()
        self.log.info(
            "coaddedmap.base_map.built",
            map_start_time=base_map.observation_start,
            map_end_time=base_map.observation_end,
            map_frequency=base_map.frequency,
        )

        self.rho = base_map.rho.copy()
        self.kappa = base_map.kappa.copy()
        self.map_resolution = np.abs(base_map.rho.wcs.wcs.cdelt[0]) * u.degrees

        self.get_time_and_mapdepth(base_map)
        self.update_map_times(base_map)
        self.initialized = True
        self.log.info("coaddedmap.coadd.initialized")

        if len(self.source_maps) == 1:
            self.log.warning("coaddedmap.coadd.single_map_warning", n_maps_coadded=1)
            self.n_maps = 1
            return

        for sourcemap in self.source_maps[1:]:
            sourcemap.build()
            sourcemap.finalize()
            self.log.info(
                "coaddedmap.source_map.built",
                map_start_time=sourcemap.observation_start,
                map_end_time=sourcemap.observation_end,
                map_frequency=sourcemap.frequency,
            )
            self.rho = enmap.map_union(
                self.rho,
                sourcemap.rho,
            )
            self.kappa = enmap.map_union(
                self.kappa,
                sourcemap.kappa,
            )
            self.get_time_and_mapdepth(sourcemap)
            self.update_map_times(sourcemap)

        if self.time_mean is not None:
            with np.errstate(divide="ignore"):
                self.time_mean /= self.map_depth
        self.n_maps = len(self.input_map_times)
        self.log.info(
            "coaddedmap.coadd.finalized",
            n_maps_coadded=self.n_maps,
            coadd_start_time=self.observation_start,
            coadd_end_time=self.observation_end,
        )
        return

    def get_time_and_mapdepth(self, new_map):
        if isinstance(new_map.time_mean, enmap.ndmap):
            if isinstance(self.time_mean, type(None)):
                self.time_mean = (
                    new_map.time_mean.copy() + new_map.start_time.timestamp()
                )
                self.map_depth = new_map.time_mean.copy()
                self.map_depth[self.map_depth > 0] = 1.0
            else:
                self.time_mean = enmap.map_union(
                    self.time_mean,
                    new_map.time_mean + new_map.start_time.timestamp(),
                )
                new_map_depth = new_map.time_mean.copy()
                new_map_depth[new_map_depth > 0] = 1.0
                self.map_depth = enmap.map_union(
                    self.map_depth,
                    new_map_depth,
                )
        ## get earliest start time per pixel ... can I use map_union with a<b or b<a or something?

        ## get latest end time per pixel ... can I use map_union with a>b or b>a or something?

    def update_map_times(self, new_map):
        if self.observation_start is None:
            self.observation_start = new_map.start_time.timestamp()
        else:
            self.observation_start = min(
                self.observation_start, new_map.start_time.timestamp()
            )
        if self.observation_end is None:
            self.observation_end = new_map.end_time.timestamp()
        else:
            self.observation_end = max(
                self.observation_end, new_map.end_time.timestamp()
            )
        self.input_map_times += [
            0.5 * (new_map.start_time.timestamp() + new_map.end_time.timestamp())
        ]

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
        super().finalize()
