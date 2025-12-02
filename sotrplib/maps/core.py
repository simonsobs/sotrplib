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
    time_end: ndmap
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
            self.time_end[x, y] if self.time_end is not None else self.observation_end
        )
        return t_start, t_mean, t_end

    @abstractmethod
    def finalize(self):
        """
        Called just before source injection to ensure that the snr, flux, and
        time maps are available.

        apply mask if present.
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
        self.time_end = time_map
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
                float(np.amax(self.time_end)), tz=timezone.utc
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

    def finalize(self):
        self.snr = self.get_snr()
        self.flux = self.get_flux()
        if self.mask is not None:
            self.snr *= self.mask
            self.flux *= self.mask
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
                float(np.amax(self.time_end)), tz=timezone.utc
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

    def finalize(self):
        self.snr = self.get_snr()
        self.flux = self.get_flux()
        if self.mask is not None:
            self.snr *= self.mask
            self.flux *= self.mask
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
        self.time_end = time_map
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
                float(np.amax(self.time_end)), tz=timezone.utc
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

    def finalize(self):
        self.snr = self.get_snr()
        self.flux = self.get_flux()
        if self.mask is not None:
            self.snr *= self.mask
            self.flux *= self.mask
        super().finalize()


"""
Need to update this to use individual time maps and calcaulte the time_start, time_mean and time_end maps.
"""


class CoaddedRhoKappaMap(ProcessableMap):
    """ """

    def __init__(
        self,
        input_maps: list[ProcessableMap],
        frequency: str,
        map_depth: ndmap | None = None,
        array: str | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.input_maps = input_maps
        self.map_depth = map_depth
        self.frequency = frequency
        self.array = array if array is not None else "coadd"

        self.input_map_times: list[float] = []
        self.time_end = None
        self.time_first = None
        self.time_mean = None
        self.observation_start = None
        self.observation_end = None

        self.log = log or structlog.get_logger()

    def build(self):
        ## check if frequency and arrays match
        good_maps = [True] * len(self.input_maps)
        for imap in self.input_maps:
            if self.frequency != imap.frequency:
                good_maps[self.input_maps.index(imap)] = False
            if self.array is not None and self.array != imap.array:
                good_maps[self.input_maps.index(imap)] = False

        if not any(good_maps):
            self.log.warning(
                "coaddedrhokappamap.coadd.no_good_maps",
                n_input_maps=len(self.input_maps),
                frequency=self.frequency,
                array=self.array,
            )
            return
        if not all(good_maps):
            self.log.info(
                "coaddedrhokappamap.coadd.dropping_maps",
                n_input_maps=len(self.input_maps),
                n_dropped_maps=len([good for good in good_maps if not good]),
                frequency=self.frequency,
                array=self.array,
            )
        self.input_maps = [
            imap for imap, good in zip(self.input_maps, good_maps) if good
        ]
        base_map = self.input_maps[0]
        base_map.build()
        self.log.info(
            "coaddedrhokappamap.base_map.built",
            map_start_time=base_map.observation_start,
            map_end_time=base_map.observation_end,
            map_frequency=base_map.frequency,
        )

        self.rho = enmap.enmap(base_map.rho)
        self.kappa = enmap.enmap(base_map.kappa)

        self.map_resolution = u.Quantity(
            abs(base_map.rho.wcs.wcs.cdelt[0]), base_map.rho.wcs.wcs.cunit[0]
        )
        self.flux_units = base_map.flux_units

        self.get_time_and_mapdepth(base_map)
        self.update_map_times(base_map)
        self.initialized = True
        self.log.info("coaddedrhokappamap.coadd.initialized")

        if len(self.input_maps) == 1:
            self.log.warning(
                "coaddedrhokappamap.coadd.single_map_warning", n_maps_coadded=1
            )
            self.n_maps = 1
            return

        for sourcemap in self.input_maps[1:]:
            sourcemap.build()
            self.log.info(
                "coaddedrhokappamap.source_map.built",
                map_start_time=sourcemap.observation_start,
                map_end_time=sourcemap.observation_end,
                map_frequency=sourcemap.frequency,
            )
            ## will want to do a weighted sum using inverse variance.
            if sourcemap.flux_units != self.flux_units:
                flux_conv = u.Quantity(1.0, sourcemap.flux_units).to(self.flux_units)
                sourcemap.rho *= flux_conv
                sourcemap.kappa /= flux_conv * flux_conv
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
                self.time_mean = enmap.enmap(new_map.time_mean)
                self.map_depth = enmap.enmap(new_map.time_mean)
                self.map_depth[self.map_depth > 0] = 1.0
            else:
                self.time_mean = enmap.map_union(
                    self.time_mean,
                    new_map.time_mean,
                )
                new_map_depth = enmap.enmap(new_map.time_mean)
                new_map_depth[new_map_depth > 0] = 1.0
                self.map_depth = enmap.map_union(
                    self.map_depth,
                    new_map_depth,
                )
        else:
            self.log.error(
                "coaddedrhokappamap.get_time_and_mapdepth.no_time_mean",
            )
        ## get earliest start time per pixel ... can I use map_union with a<b or b<a or something?
        if isinstance(new_map.time_first, ndmap):
            if isinstance(self.time_first, type(None)):
                self.time_first = enmap.enmap(new_map.time_first)
            else:
                self.time_first = np.minimum(
                    self.time_first,
                    new_map.time_first,
                )
        else:
            self.log.error(
                "coaddedrhokappamap.get_time_and_mapdepth.no_time_first",
            )
        ## get latest end time per pixel ... can I use map_union with a>b or b>a or something?
        if isinstance(new_map.time_end, ndmap):
            if isinstance(self.time_end, type(None)):
                self.time_end = enmap.enmap(new_map.time_end)
            else:
                self.time_end = np.maximum(
                    self.time_end,
                    new_map.time_end,
                )
        else:
            self.log.error(
                "coaddedrhokappamap.get_time_and_mapdepth.no_time_end",
            )

    def update_map_times(self, new_map):
        if self.observation_start is None:
            self.observation_start = new_map.observation_start
        else:
            self.observation_start = min(
                self.observation_start, new_map.observation_start
            )
        if self.observation_end is None:
            self.observation_end = new_map.observation_end
        else:
            self.observation_end = max(self.observation_end, new_map.observation_end)
        time_delta = new_map.observation_end - new_map.observation_start
        mid_time = new_map.observation_start + (time_delta / 2)
        self.input_map_times.append(mid_time)

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

    def finalize(self):
        self.snr = self.get_snr()
        self.flux = self.get_flux()
        self.time_mean /= self.map_depth
        super().finalize()
