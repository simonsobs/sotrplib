"""
Core map objects.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path

import astropy.units as u
import numpy as np
import structlog
from astropy.coordinates import SkyCoord
from astropy.units import Unit
from astropydantic import AstroPydanticQuantity
from pixell import enmap
from pixell.enmap import ndmap
from pydantic import AwareDatetime
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
    time_last: ndmap
    "The time at which each pixel was last observed"
    time_mean: ndmap
    "The mean time at which each pixel was observed"
    hits: ndmap
    "A hits map stating the number of times each pixel was observed"
    mask: ndmap | None = None
    "A mask map indicating valid pixels as 1 and invalid pixels as 0"

    finalized: bool = False
    "Whether finalize has been called and ancillary maps can no longer be updated"

    frequency: str
    "The frequency band of the map, e.g. f090"
    array: str
    "The array/wafer that was used"

    observation_length: timedelta
    "Total length of the observation"
    observation_start: AwareDatetime
    "Start time of the observation"
    observation_end: AwareDatetime
    "End time of the observation"
    observation_time: AwareDatetime
    "Rough 'middle' time of the observation"

    flux_units: Unit
    map_resolution: u.Quantity | None

    instrument: str | None = None
    "The instrument that observed the map, e.g. 'SOLAT', 'SOSAT'"

    box: tuple[SkyCoord, SkyCoord] | None = None
    __rho: ndmap | None = None
    __kappa: ndmap | None = None

    _map_id: str | None = None
    "An identifier for the map, e.g. filename or coadd type"

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
        ## TODO: do we want the bounding box returned to be the bbox of this specific map
        ## or the box used to read in the map if it was provided?
        ## need to check how box works when reading in the maps;
        """
        let's say you want to cut a box from -20 to 20 in ra, and -20 to 20 in dec but the map is only from -10, 20 in ra and -20, 10 in dec. 
        The true bounding box of the cut map would only be (-10,-20),(20,10) but self.box would be (-20,-10),(20,20)
        not sure if that's how box works when reading in maps or if it fills the empty space with nans or something... should check that
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

    @property
    def map_id(self) -> str:
        """
        An identifier for the map, e.g. filename or coadd type
        Defaults to {frequency}_{array}_{observationstart_timestamp}
        """
        return (
            self._map_id
            if self._map_id
            else f"{self.frequency}_{self.array}_{int(self.observation_start.timestamp())}"
        )

    @abstractmethod
    def get_pixel_times(self, pix: tuple[int, int]):
        """
        Given a pixel in the map, return the observation start, mean and end time of that pixel.
        """
        x, y = int(pix[0]), int(pix[1])

        t_start = (
            self.time_first[x, y]
            if self.time_first is not None
            else self.observation_start
        )
        t_mean = (
            self.time_mean[x, y]
            if self.time_mean is not None
            else self.observation_time
        )
        t_end = (
            self.time_last[x, y] if self.time_last is not None else self.observation_end
        )
        return t_start, t_mean, t_end

    @abstractmethod
    def apply_mask(self):
        """
        Apply the mask to the snr and flux maps if a mask is present.
        Assumes masked region is indicated by 0s in the mask map.

        """
        if self.mask is not None:
            self.snr *= self.mask
            self.flux *= self.mask
        return

    @abstractmethod
    def finalize(self):
        """
        Called just before source injection to ensure that the snr, flux, and
        time maps are available.

        apply mask if present.
        """
        del self.rho
        del self.kappa
        self.apply_mask()
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
        start_time: AwareDatetime,
        end_time: AwareDatetime,
        box: AstroPydanticQuantity[u.deg] | None = None,
        time_filename: Path | None = None,
        info_filename: Path | None = None,
        frequency: str | None = None,
        array: str | None = None,
        instrument: str | None = None,
        matched_filtered: bool = False,
        mask: ndmap | None = None,
        intensity_units: Unit = u.K,
        log: FilteringBoundLogger | None = None,
    ):
        self.intensity_filename = intensity_filename
        self.inverse_variance_filename = inverse_variance_filename
        self.intensity_units = intensity_units
        self.time_filename = time_filename
        self.info_filename = info_filename
        self.observation_start = start_time
        self.observation_end = end_time
        self.box = box
        self.frequency = frequency
        self.array = array
        self.instrument = instrument
        self.matched_filtered = matched_filtered
        self.mask = mask
        self.log = log or structlog.get_logger()

    def build(self):
        log = self.log.bind(intensity_filename=self.intensity_filename)
        box = self.box.to(u.rad).value if self.box is not None else None
        intensity_shape = enmap.read_map_geometry(str(self.intensity_filename))[0]
        self.intensity = enmap.read_map(
            str(self.intensity_filename),
            sel=0 if len(intensity_shape) > 2 else None,
            box=box,
        )
        log.debug("intensity_ivar.intensity.read")

        # TODO: Set metadata from header e.g. frequency band.
        log = log.new(inverse_variance_filename=self.inverse_variance_filename)
        inverse_variance_shape = enmap.read_map_geometry(
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
            time_shape = enmap.read_map_geometry(str(self.time_filename))[0]
            time_map = enmap.read_map(
                str(self.time_filename), sel=0 if len(time_shape) > 2 else None, box=box
            )
            log.debug("intensity_ivar.time.read")
        else:
            time_map = None
            log.debug("intensity_ivar.time.none")

        ##TODO : what to do here?
        self.time_first = time_map
        self.time_last = time_map
        self.time_mean = time_map
        self.add_time_offset(self.observation_start)
        self.observation_length = self.observation_end - self.observation_start

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
        if offset is None:
            from pixell.bunch import read as bunch_read

            offset = (
                datetime.fromtimestamp(
                    bunch_read(str(self.info_filename)).t, tz=timezone.utc
                )
                if self.info_filename is not None
                else None
            )
        if offset is None:
            self.log.warning("intensity_ivar.time_offset.none")
            return
        self.observation_start = offset
        ## TODO: time_first, mean, end are same so this changes all.
        ## this should be changed when we do something smarter.
        if self.time_first is not None:
            self.time_first[self.time_first > 0] += offset.timestamp()
        if self.observation_end is None:
            self.observation_end = datetime.fromtimestamp(
                float(np.amax(self.time_last)), tz=timezone.utc
            )

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

    def get_pixel_times(self, pix: tuple[int, int]):
        return super().get_pixel_times(pix)

    def apply_mask(self):
        return super().apply_mask()

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
        self.time_last = self.prefiltered_map.time_last
        self.time_mean = self.prefiltered_map.time_mean
        self.observation_start = self.prefiltered_map.observation_start
        self.observation_end = self.prefiltered_map.observation_end
        self.observation_length = (
            self.prefiltered_map.observation_end
            - self.prefiltered_map.observation_start
        )
        self.box = self.prefiltered_map.box
        self.frequency = self.prefiltered_map.frequency
        self.array = self.prefiltered_map.array
        self.mask = self.prefiltered_map.mask
        self.instrument = self.prefiltered_map.instrument

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
        if offset is None:
            from pixell.bunch import read as bunch_read

            offset = (
                datetime.fromtimestamp(
                    bunch_read(str(self.info_filename)).t, tz=timezone.utc
                )
                if self.info_filename is not None
                else None
            )
        if offset is None:
            self.log.warning("matchedfiltered_intensity_ivar.time_offset.none")
            return
        self.observation_start = offset
        ## TODO: time_first, mean, end are same so this changes all.
        ## this should be changed when we do something smarter.
        if self.time_first is not None:
            self.time_first[self.time_first > 0] += offset.timestamp()
        if self.observation_end is None:
            self.observation_end = datetime.fromtimestamp(
                float(np.amax(self.time_last)), tz=timezone.utc
            )

    def get_snr(self):
        with np.errstate(divide="ignore"):
            snr = self.rho / np.sqrt(self.kappa)

        return snr

    def get_flux(self):
        with np.errstate(divide="ignore"):
            flux = self.rho / self.kappa

        return flux

    def get_pixel_times(self, pix: tuple[int, int]):
        return super().get_pixel_times(pix)

    def apply_mask(self):
        return super().apply_mask()

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
        start_time: AwareDatetime,
        end_time: AwareDatetime,
        box: AstroPydanticQuantity[u.deg] | None = None,
        time_filename: Path | None = None,
        info_filename: Path | None = None,
        frequency: str | None = None,
        array: str | None = None,
        instrument: str | None = None,
        flux_units: Unit = u.Jy,
        mask: ndmap | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.rho_filename = rho_filename
        self.kappa_filename = kappa_filename
        self.time_filename = time_filename
        self.info_filename = info_filename
        self.observation_start = start_time
        self.observation_end = end_time
        self.box = box
        self.frequency = frequency
        self.array = array
        self.instrument = instrument
        self.flux_units = flux_units
        self.mask = mask
        self.log = log or structlog.get_logger()

    def build(self):
        log = self.log.bind(rho_filename=self.rho_filename)
        box = self.box.to(u.rad).value if self.box is not None else None
        try:
            self.rho = enmap.read_map(str(self.rho_filename), sel=0, box=box)
            assert type(self.rho) is enmap.ndmap
        except (IndexError, AttributeError, AssertionError):
            # Rho map does not have Q, U
            self.rho = enmap.read_map(str(self.rho_filename), box=box)

        log = log.new(kappa_filename=self.kappa_filename)
        try:
            self.kappa = enmap.read_map(str(self.kappa_filename), sel=0, box=box)
            assert type(self.kappa) is enmap.ndmap
        except (IndexError, AttributeError, AssertionError):
            # Kappa map does not have Q, U
            self.kappa = enmap.read_map(str(self.kappa_filename), box=box)

        self.map_resolution = u.Quantity(
            abs(self.rho.wcs.wcs.cdelt[0]), self.rho.wcs.wcs.cunit[0]
        )

        log = log.new(time_filename=self.time_filename)
        if self.time_filename is not None:
            # TODO: Handle nuance that the start time is not included.
            try:
                time_map = enmap.read_map(str(self.time_filename), sel=0, box=box)
                assert type(time_map) is enmap.ndmap
            except (IndexError, AttributeError, AssertionError):
                # Somehow time map requires sel=0
                time_map = enmap.read_map(str(self.time_filename), box=box)
        else:
            time_map = None
            log.debug("rho_kappa.time.none")

        self.time_first = time_map
        self.time_last = time_map
        self.time_mean = time_map

        self.add_time_offset(self.observation_start)
        self.observation_length = self.observation_end - self.observation_start
        return

    def add_time_offset(self, offset: timedelta | None = None):
        """
        Add a time offset to the time maps. Useful if you have a time map
        that is relative to the start of the observation, and you want to
        convert it to an absolute time map.

        time maps are in unix time (seconds).

        ACT time maps are stored in seconds since the start of the observation,
        thus if you want absolute time you need to add the start time of the observation.
        """
        if offset is None:
            from pixell.bunch import read as bunch_read

            offset = (
                datetime.fromtimestamp(
                    bunch_read(str(self.info_filename)).t, tz=timezone.utc
                )
                if self.info_filename is not None
                else None
            )
        if offset is None:
            self.log.warning("rho_kappa.time_offset.none")
            return
        self.observation_start = offset
        ## TODO: time_first, mean, end are same so this changes all.
        ## this should be changed when we do something smarter.
        if self.time_first is not None:
            self.time_first[self.time_first > 0] += offset.timestamp()
        if self.observation_end is None:
            self.observation_end = datetime.fromtimestamp(
                float(np.amax(self.time_last)), tz=timezone.utc
            )

    def get_pixel_times(self, pix: tuple[int, int]):
        return super().get_pixel_times(pix)

    def get_snr(self):
        with np.errstate(divide="ignore"):
            snr = self.rho / np.sqrt(self.kappa)
        return snr

    def get_flux(self):
        with np.errstate(divide="ignore"):
            flux = self.rho / self.kappa
        return flux

    def apply_mask(self):
        return super().apply_mask()

    def finalize(self):
        self.snr = self.get_snr()
        self.flux = self.get_flux()
        super().finalize()


class CoaddedRhoKappaMap(ProcessableMap):
    """
    Docstring for CoaddedMap
    """

    input_map_list: list[str]

    def __init__(
        self,
        rho: ndmap,
        kappa: ndmap,
        observation_start: AwareDatetime,
        observation_end: AwareDatetime,
        time_first: ndmap | None = None,
        time_mean: ndmap | None = None,
        time_last: ndmap | None = None,
        observation_length: timedelta | None = None,
        box: AstroPydanticQuantity[u.deg] | None = None,
        frequency: str | None = None,
        array: str | None = None,
        instrument: str | None = None,
        flux_units: Unit = u.Jy,
        mask: ndmap | None = None,
        map_resolution: u.Quantity | None = None,
        hits: ndmap | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.rho = rho
        self.kappa = kappa
        self.time_first = time_first
        self.time_mean = time_mean
        self.time_last = time_last
        self.observation_start = observation_start
        self.observation_end = observation_end
        self.observation_length = observation_length
        self.box = box
        self.frequency = frequency
        self.array = array
        self.instrument = instrument
        self.flux_units = flux_units
        self.mask = mask
        self.hits = hits
        self.map_resolution = map_resolution
        self.log = log or structlog.get_logger()

    def build(self):
        pass

    def get_pixel_times(self, pix: tuple[int, int]):
        return super().get_pixel_times(pix)

    def get_snr(self):
        with np.errstate(divide="ignore"):
            snr = self.rho / np.sqrt(self.kappa)
        return snr

    def get_flux(self):
        with np.errstate(divide="ignore"):
            flux = self.rho / self.kappa
        return flux

    def apply_mask(self):
        return super().apply_mask()

    def finalize(self):
        self.snr = self.get_snr()
        self.flux = self.get_flux()
        super().finalize()
