"""
Core map objects.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import astropy.units as u
import numpy as np
import structlog
from astropy.coordinates import SkyCoord
from astropy.units import Unit
from astropydantic import AstroPydanticQuantity
from pixell import enmap
from pixell.enmap import ndmap
from structlog.types import FilteringBoundLogger


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

    box: tuple[SkyCoord, SkyCoord] | None = None
    __rho: ndmap | None = None
    __kappa: ndmap | None = None

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
                if (x := getattr(self, attribute, None)) is not None:
                    self.map_resolution = u.Quantity(
                        abs(x.wcs.wcs.cdelt[0]), x.wcs.wcs.cunit[0]
                    )
                    break

        return self.map_resolution

    @property
    def bbox(self) -> tuple[SkyCoord, SkyCoord]:
        """
        The bounding box of the map provided as sky coordinates.
        """
        for attribute in ["flux", "snr", "rho", "kappa"]:
            if (x := getattr(self, attribute, None)) is not None:
                shape = x.shape[-2:]
                wcs = x.wcs

                bottom_left = wcs.array_index_to_world(0, 0)
                top_right = wcs.array_index_to_world(shape[0], shape[1])

                self.box = (
                    bottom_left,
                    top_right,
                )

                break

        return self.box

    @abstractmethod
    def finalize(self):
        """
        Called just before source injection to ensure that the snr, flux, and
        time maps are available.
        """
        del self.rho
        del self.kappa

        self.finalized = True

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
        box: AstroPydanticQuantity[u.deg] | None = None,
        time_filename: Path | None = None,
        frequency: str | None = None,
        array: str | None = None,
        matched_filtered: bool = False,
        intensity_units: Unit = u.K,
        log: FilteringBoundLogger | None = None,
    ):
        self.intensity_filename = intensity_filename
        self.inverse_variance_filename = inverse_variance_filename
        self.intensity_units = intensity_units
        self.time_filename = time_filename
        self.observation_start = start_time
        self.observation_end = end_time
        self.box = box
        self.frequency = frequency
        self.array = array
        self.matched_filtered = matched_filtered
        self.log = log or structlog.get_logger()

    def build(self):
        log = self.log.bind(intensity_filename=self.intensity_filename)
        box = self.box.to(u.rad).value if self.box is not None else None

        intensity_shape = enmap.read_geometry(str(self.intensity_filename))[0]
        self.intensity = enmap.read_map(
            str(self.intensity_filename),
            sel=0 if len(intensity_shape) > 2 else None,
            box=box,
        )
        log.debug("intensity_ivar.intensity.read")

        # TODO: Set metadata from header e.g. frequency band.
        log = log.new(inverse_variance_filename=self.inverse_variance_filename)
        inverse_variance_shape = enmap.read_geometry(
            str(self.inverse_variance_filename)
        )[0]
        self.inverse_variance = enmap.read_map(
            str(self.inverse_variance_filename),
            sel=0 if len(inverse_variance_shape) > 2 else None,
            box=box,
        )
        log.debug("intensity_ivar.ivar.read")

        self.map_resolution = u.Quantity(
            abs(self.inverse_variance.wcs.wcs.cdelt[0]),
            self.inverse_variance.wcs.wcs.cunit[0],
        )

        log = log.new(time_filename=self.time_filename)
        if self.time_filename is not None:
            # TODO: Handle nuance that the start time is not included.
            time_shape = enmap.read_geometry(str(self.time_filename))[0]
            time_map = enmap.read_map(
                str(self.time_filename), sel=0 if len(time_shape) > 2 else None, box=box
            )
            log.debug("intensity_ivar.time.read")
        else:
            time_map = None
            log.debug("intensity_ivar.time.none")

        self.time_first = time_map
        self.time_end = time_map
        self.time_mean = time_map

        return

    def get_snr(self):
        with np.errstate(divide="ignore"):
            snr = self.intensity / np.sqrt(self.inverse_variance)
        return snr

    def get_flux(self):
        with np.errstate(divide="ignore"):
            flux = self.intensity
        ## assume that if matched filtered, intensity is rho and ivar is kappa
        if self.matched_filtered:
            flux /= self.inverse_variance
        return flux

    def finalize(self):
        self.snr = self.get_snr()
        self.flux = self.get_flux()
        super().finalize()


class MatchedFilteredIntensityAndInverseVarianceMap(ProcessableMap):
    """
    Resulting rho/kappa maps after ingesting intensity and ivar maps and
    matched filtering them. Could be Depth 1, could be monthly
    or weekly co-adds. Or something else!

    """

    def __init__(
        self,
        rho: enmap.ndmap,
        kappa: enmap.ndmap,
        flux_units: u.Unit,
        prefiltered_map: IntensityAndInverseVarianceMap,
        keep_prefiltered: bool = False,
        log: FilteringBoundLogger | None = None,
    ):
        self.rho = rho
        self.kappa = kappa
        self.prefiltered_map = prefiltered_map
        self.flux_units = flux_units
        self.log = log or structlog.get_logger()
        self.time_first = self.prefiltered_map.time_first
        self.time_end = self.prefiltered_map.time_end
        self.time_mean = self.prefiltered_map.time_mean
        self.observation_start = self.prefiltered_map.observation_start
        self.observation_end = self.prefiltered_map.observation_end
        self.box = self.prefiltered_map.box
        self.frequency = self.prefiltered_map.frequency
        self.array = self.prefiltered_map.array

        self.map_resolution = u.Quantity(
            abs(self.rho.wcs.wcs.cdelt[0]), self.rho.wcs.wcs.cunit[0]
        )
        if keep_prefiltered:
            self.prefiltered_intensity = self.prefiltered_map.intensity
            self.prefiltered_inverse_variance = self.prefiltered_map.inverse_variance
            self.intensity_units = self.prefiltered_map.intensity_units
        else:
            self.prefiltered_map = None
            self.prefiltered_intensity = None
            self.prefiltered_inverse_variance = None
            self.intensity_units = None

    def build(self):
        return

    def add_time_offset(self, offset: timedelta):
        """
        Add a time offset to the time maps. Useful if you have a time map
        that is relative to the start of the observation, and you want to
        convert it to an absolute time map.

        time maps are in unix time (seconds).

        ACT time maps are stored in seconds since the start of the observation,
        thus if you want absolute time you need to add the start time of the observation.
        """
        if self.time_first is not None:
            self.time_first += offset.timestamp()
        if self.time_end is not None:
            self.time_end += offset.timestamp()
        if self.time_mean is not None:
            self.time_mean += offset.timestamp()

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


class RhoAndKappaMap(ProcessableMap):
    """
    A set of FITS maps read from disk. Could be Depth 1, could
    be monthly or weekly co-adds. Or something else!

    box is array of astropy Quantities: [[dec_min, ra_min], [dec_max, ra_max]]
    """

    def __init__(
        self,
        rho_filename: Path,
        kappa_filename: Path,
        start_time: datetime,
        end_time: datetime,
        box: AstroPydanticQuantity[u.deg] | None = None,
        time_filename: Path | None = None,
        frequency: str | None = None,
        array: str | None = None,
        flux_units: Unit = u.Jy,
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
        self.flux_units = flux_units
        self.log = log or structlog.get_logger()

    def build(self):
        log = self.log.bind(rho_filename=self.rho_filename)
        box = self.box.to(u.rad).value if self.box is not None else None
        try:
            self.rho = enmap.read_map(str(self.rho_filename), sel=0, box=box)
            log.debug("rho_kappa.rho.read.sel")
        except (IndexError, AttributeError):
            # Rho map does not have Q, U
            self.rho = enmap.read_map(str(self.rho_filename), box=box)
            log.debug("rho_kappa.rho.read.nosel")

        log = log.new(kappa_filename=self.kappa_filename)
        try:
            self.kappa = enmap.read_map(str(self.kappa_filename), sel=0, box=box)
            log.debug("rho_kappa.kappa.read.sel")
        except (IndexError, AttributeError):
            # Kappa map does not have Q, U
            self.kappa = enmap.read_map(str(self.kappa_filename), box=box)
            log.debug("rho_kappa.kappa.read.nosel")

        self.map_resolution = u.Quantity(
            abs(self.rho.wcs.wcs.cdelt[0]), self.rho.wcs.wcs.cunit[0]
        )

        log = log.new(time_filename=self.time_filename)
        if self.time_filename is not None:
            # TODO: Handle nuance that the start time is not included.
            try:
                time_map = enmap.read_map(str(self.time_filename), sel=0, box=box)
                log.debug("rho_kappa.time.read.sel")
            except (IndexError, AttributeError):
                # Somehow time map requires sel=0
                time_map = enmap.read_map(str(self.time_filename), box=box)
                log.debug("rho_kappa.time.read.nosel")
        else:
            time_map = None
            log.debug("rho_kappa.time.none")

        self.time_first = time_map
        self.time_end = time_map
        self.time_mean = time_map

        return

    def add_time_offset(self, offset: timedelta):
        """
        Add a time offset to the time maps. Useful if you have a time map
        that is relative to the start of the observation, and you want to
        convert it to an absolute time map.

        time maps are in unix time (seconds).

        ACT time maps are stored in seconds since the start of the observation,
        thus if you want absolute time you need to add the start time of the observation.
        """
        if self.time_first is not None:
            self.time_first += offset.timestamp()
        if self.time_end is not None:
            self.time_end += offset.timestamp()
        if self.time_mean is not None:
            self.time_mean += offset.timestamp()

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
        time_start: ndmap | None = None,
        time_mean: ndmap | None = None,
        time_end: ndmap | None = None,
        map_depth: ndmap | None = None,
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
        self.map_resolution = u.Quantity(
            abs(base_map.rho.wcs.wcs.cdelt[0]), base_map.rho.wcs.wcs.cunit[0]
        )

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
        if isinstance(new_map.time_mean, ndmap):
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
