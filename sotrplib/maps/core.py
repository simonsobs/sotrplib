"""
Core map objects.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import astropy.units as u
import numpy as np
import structlog
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from astropy.units import Unit
from mapcat.pointing.const import ConstantPointingModel
from pixell import enmap
from pixell.enmap import ndmap
from structlog.types import FilteringBoundLogger

from sotrplib.maps.utils import (
    enmap_box_to_skycoord,
    pixell_map_union,
    skycoord_box_to_enmap_box,
)

PointingModel = ConstantPointingModel


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
    mask: ndmap | None = None
    "A mask map indicating valid pixels as 1 and invalid pixels as 0"
    sky_model: ndmap | None = None
    "A model of the sky - either a coadd or a model built after measuring sources."
    "Can be used for source subtraction."
    flatfield_map: ndmap | None = None
    "A flatfield map containing the local 2D background RMS used for flatfielding the map, if applicable."

    finalized: bool = False
    "Whether finalize has been called and ancillary maps can no longer be updated"

    frequency: str
    "The frequency band of the map, e.g. f090"
    array: str
    "The array/wafer that was used"

    observation_length: TimeDelta
    "Total length of the observation"
    observation_start: Time
    "Start time of the observation"
    observation_end: Time
    "End time of the observation"
    observation_time: Time
    "Rough 'middle' time of the observation"

    flux_units: Unit
    map_resolution: u.Quantity | None

    instrument: str | None = None
    "The instrument that observed the map, e.g. 'SOLAT', 'SOSAT'"

    _sky_box: tuple[SkyCoord, SkyCoord] | None = None

    _hits: ndmap | None = None
    "A hits map stating the number of times each pixel was observed"

    _valid_pixel_mask_cache: ndmap | None = None

    __rho: ndmap | None = None
    __kappa: ndmap | None = None

    __map_id: str | None = None
    "An identifier for the map, e.g. filename or coadd type"

    _parent_database: Path | None = None
    "Path to the parent database for this map, if any"

    _pointing_model: PointingModel | None = None

    _available_maps: tuple[str, ...] = ("flux", "snr")
    "Ordered attribute names searched by bbox to find a loaded map."

    @abstractmethod
    def build(self):
        """
        Build the maps for snr, flux, and time. Could include simulating them,
        or reading them from disk.
        """
        return

    @abstractmethod
    def _compute_hits(self):
        """
        Must return an ndmap of hits.
        """
        raise NotImplementedError(
            "_compute_hits must be implemented by ProcessableMap subclass"
        )

    def _core_filter_sources(self, source_positions: SkyCoord, bool_map: enmap.ndmap):
        """
        Core method for filtering sources. Must be implemented by ProcessableMap subclass.
        Takes in a list of sources and a boolean map, and returns the list of sources within the valid pixels.
        Bool map should have 1 or True in valid region and 0 or False in unobserved (or masked) region.
        """
        fvals = bool_map.at(
            (source_positions.dec.to_value("rad"), source_positions.ra.to_value("rad")),
            mode="nn",
        )
        result = (np.isfinite(fvals)) & (fvals > 0.0)
        return result, source_positions[result]

    @abstractmethod
    def _compute_valid_pixel_mask(self) -> enmap.ndmap:
        raise NotImplementedError(
            "_compute_valid_pixel_mask must be implemented by ProcessableMap subclass"
        )

    def get_valid_pixel_mask(self) -> enmap.ndmap:
        if self._valid_pixel_mask_cache is None:
            self._valid_pixel_mask_cache = self._compute_valid_pixel_mask()
        return self._valid_pixel_mask_cache

    def filter_sources(self, source_positions: SkyCoord):
        """
        Filter sources based on the mask and hits map. Returns the positions of sources that are in valid pixels.
        """
        return self._core_filter_sources(source_positions, self.get_valid_pixel_mask())

    def get_kappa_thresholds(self):
        kappa_map = self.kappa
        print("get kappa thresh load in")

        valid = np.isfinite(kappa_map) & (kappa_map > 0)

        vals = kappa_map[valid]

        n = len(vals)

        p1 = int(0.01 * (n - 1))
        p_frac = int(0.025 * (n - 1))
        p_frac2 = int(0.05 * (n - 1))

        part = np.partition(vals, [p1, p_frac, p_frac2])

        kappa_thresh = part[
            p1
        ]  # value below which only 1% of pixels lie (lowest 1% of kappa so highest uncertainties)

        frac_kappa_thresh10 = part[
            p_frac
        ]  # worst 2.5% pixels on the depth-1 map (threshold for 10x10 thumbnail)

        frac_kappa_thresh40 = part[
            p_frac2
        ]  # worst 5% pixels on the depth-1 map (threshold for 40x40 thumbnail)
        print("min =", np.min(vals))
        print("max =", np.max(vals))
        print("p1 =", p1)
        print("threshold =", part[p1])

        return (kappa_thresh, frac_kappa_thresh10, frac_kappa_thresh40)

    ######### rejection function using kappa thresholds
    def reject_badmaps(
        self,
        source_positions: SkyCoord,
        time,
        kappa_thresh,
        frac_kappa_thresh10,
        frac_kappa_thresh40,
        frac_low_10x10=0.25,
        frac_low_40x40=0.5,
    ):
        print("REJECT FUNCTION CALLED")

        kappa_map = self._kappa_for_rejection
        print("reject function load in")
        # sky coordinates to pixel indices
        pos = (
            source_positions.dec.to_value("rad"),
            source_positions.ra.to_value("rad"),
        )  # of agn
        y_f, x_f = kappa_map.sky2pix(
            pos
        )  # y,x pixel coordinates (to find in depth1 maps)
        y, x = (
            int(y_f),
            int(x_f),
        )  # for thumbnailing maps later on in this function (array slicing)

        # basically, the 1% of pixels with the lowest inverse variance in the map are treated as candidates for rejection
        # since uncertainty and kappa are inversely proportional, higher kappa -> lower uncertainty whereas lower kappa -> higher uncertainty. So, since we're taking the lowest 1% of kappa here (1st percentile), then only the worst and most unreliable values (that have large uncertainties) are considered the threshold.

        # central pixel (point source)
        k_center = kappa_map.at(pos, mode="nn")
        print("SOURCE:", source_positions, "k_center:", k_center)
        print(f"threshold:{kappa_thresh:.12e}")

        # thumbnail slices centered on said source
        k10_thumb = kappa_map[y - 5 : y + 5, x - 5 : x + 5]  # 10x10
        k40_thumb = kappa_map[y - 20 : y + 20, x - 20 : x + 20]  # 40x40

        reject = 1  # this means that the anything in the catalog that has 1 in this column is not rejected (by default)
        reject_con = 0
        BAD_TIMES = {
            1502427600,
            1502341200,
            1502409600,
        }  # cap was left on during the observations for these times so we don't use them

        if (
            k_center <= kappa_thresh
        ):  # making sure the source is ACTUALLY at the center (equiv. to hits.at(pos) <= 10), rejects map cuz the source location itself is bad quality
            reject = 0
            reject_con = 1

        # 10x10 thumbnail
        # Note that frac_kappa_thresh10 and frac_kappa_thresh40 are defined in the process_info_files function
        elif (
            np.mean(k10_thumb <= frac_kappa_thresh10) >= frac_low_10x10
        ):  # local neighborhood contamination
            # this is basically saying that if you take 25% the pixels in this 10x10 thumbnail and their kappa (inverse variance) values are lower than the lowest 2.5% kappa values in the whole depth-1 map that this thumbnail is from, then it's got even higher noise than that worst 2.5% in the depth-1 map, and should thus be rejected
            # (idk if that made sense but I can draw a little diagram if necessary!)
            reject = 0
            reject_con = 2

        # 40x40 thumbnail
        elif (
            np.mean(k40_thumb <= frac_kappa_thresh40) >= frac_low_40x40
        ):  # large scale contamination
            reject = 0
            reject_con = 3

        # bad times
        elif int(time) in BAD_TIMES:
            reject = 0
            reject_con = 4

        # return after all checks
        return reject, reject_con

    @property
    def hits(self):
        """
        Lazily computed hits map.
        """
        if self._hits is None:
            self._hits = self._compute_hits()
        return self._hits

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
    def bbox(self) -> np.ndarray:
        """
        The bounding box of the map as a pixell enmap box.

        Returns ``enmap.box(shape, wcs)`` on the first loaded map found in
        ``_available_maps``.
        Returns ``None`` if no map or sky_box is available.
        """
        for attr in self._available_maps:
            if (m := getattr(self, attr, None)) is not None:
                return enmap.box(m.shape, m.wcs)

        return None

    @property
    def sky_box(self) -> tuple[SkyCoord, SkyCoord] | None:
        """
        Sky bounding box. Can be set to initialize the map with a cutout of a larger map.
        Should be in the form

        ``sky_box = (SkyCoord(ra=ra_min, dec=dec_min), SkyCoord(ra=ra_max, dec=dec_max))``

        Wrap detection (both RAs in ``[0, 2π)``):

        - ``sky_box[0].ra < sky_box[1].ra`` — non-wrapping region.
        - ``sky_box[0].ra > sky_box[1].ra`` — wrapping region (spans RA = 0);
          ``sky_box[0].ra`` is the upper boundary (near 2π) and ``sky_box[1].ra`` is the
          lower boundary (near 0).

        If not explicitly set, derived from ``bbox`` (e.g. the geometry of a
        loaded map) the same way ``rho``/``kappa`` are derived from ``snr``/``flux``.
        """
        if self._sky_box is not None:
            return self._sky_box
        if (box := self.bbox) is not None:
            return enmap_box_to_skycoord(box)

        return None

    @sky_box.setter
    def sky_box(self, x):
        self._sky_box = x

    @sky_box.deleter
    def sky_box(self):
        try:
            del self._sky_box
        except AttributeError:
            pass

    @property
    def map_id(self) -> str:
        """
        An identifier for the map, e.g. filename or coadd type
        Defaults to {frequency}_{array}_{observationstart_timestamp} if not set.
        """
        return self.__map_id if self.__map_id is not None else self.get_map_str_id()

    @map_id.setter
    def map_id(self, x):
        self.__map_id = x

    @map_id.deleter
    def map_id(self):
        try:
            del self.__map_id
        except AttributeError:
            pass

    def get_map_str_id(self) -> str:
        """
        Get a string identifier for the map, useful for logging.
        """
        return f"{self.frequency}_{self.array}_{int(self.observation_start.unix)}"

    def get_pixel_times(
        self, pix: tuple[int, int]
    ) -> tuple[Time | None, Time | None, Time | None]:
        """
        Given a pixel in the map, return the observation start, mean and end time of that pixel.
        Assumes time map is in unix time.

        If the map has time maps, will check pixel value. Otherwise, will
        use the time ranges defined by the map's observation start and end times.

        Raises
        ------
        IndexError
            If the pixel is outside the map bounds.

        Returns
        -------
        tuple[Time|None, Time|None, Time|None]
            The observation start, mean and end time of the pixel, or None if the pixel is outside the map bounds.
        """
        x, y = int(pix[0]), int(pix[1])

        t_start = (
            Time(float(self.time_first[x, y]), format="unix")
            if self.time_first is not None
            else Time(self.observation_start)
        )
        t_mean = (
            Time(float(self.time_mean[x, y]), format="unix")
            if self.time_mean is not None
            else self.observation_start
            + (self.observation_end - self.observation_start) / 2
        )
        t_end = (
            Time(float(self.time_last[x, y]), format="unix")
            if self.time_last is not None
            else self.observation_end
        )
        return t_start, t_mean, t_end

    def get_obs_time(self, position: SkyCoord) -> Time | None:
        """Return the per-pixel mean observation time at a sky position, or None on failure."""
        try:
            pix = self.hits.sky2pix(
                np.array([position.dec.to_value("rad"), position.ra.to_value("rad")])
            )
            _, t_pixel, _ = self.get_pixel_times(pix)
            return t_pixel
        except (IndexError, ValueError):
            return None

    def apply_mask(self):
        """
        Apply the mask to the snr and flux maps if a mask is present.
        Assumes masked region is indicated by 0s in the mask map.

        """
        if self.mask is not None:
            self.snr *= self.mask
            self.flux *= self.mask
        return

    def finalize(self):
        """
        Called just before source injection to ensure that the snr, flux, and
        time maps are available.

        clear the cached valid pixel mask because the apply mask may change that.

        apply mask if present.
        """
        del self.rho
        self.kappa_thresholds = self.get_kappa_thresholds()
        self._kappa_for_rejection = self.kappa.copy()
        del self.kappa
        self._valid_pixel_mask_cache = None
        self.apply_mask()
        self.finalized = True
        return


class IntensityAndInverseVarianceMap(ProcessableMap):
    """
    A set of FITS maps read from disk. Could be Depth 1, could
    be monthly or weekly co-adds. Or something else!
    """

    _available_maps: tuple[str, ...] = ("intensity",)

    def __init__(
        self,
        intensity_filename: Path,
        inverse_variance_filename: Path,
        start_time: Time,
        end_time: Time,
        sky_box: tuple[SkyCoord, SkyCoord] | None = None,
        time_filename: Path | None = None,
        info_filename: Path | None = None,
        frequency: str | None = None,
        array: str | None = None,
        instrument: str | None = None,
        matched_filtered: bool = False,
        mask: ndmap | None = None,
        intensity_units: Unit = u.K,
        map_id: str | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.intensity_filename = intensity_filename
        self.inverse_variance_filename = inverse_variance_filename
        self.intensity_units = intensity_units
        self.time_filename = time_filename
        self.info_filename = info_filename
        self.observation_start = start_time
        self.observation_end = end_time
        self.sky_box = sky_box
        self.frequency = frequency
        self.array = array
        self.instrument = instrument
        self.matched_filtered = matched_filtered
        self.mask = mask
        if map_id is not None:
            self.map_id = map_id
        self._hits = None
        self.log = log or structlog.get_logger()

    @property
    def bbox(self) -> np.ndarray:
        if (box := super().bbox) is not None:
            return box
        shape, wcs = enmap.read_map_geometry(str(self.intensity_filename))
        return enmap.box(shape, wcs)

    def build(self):
        log = self.log.bind(
            intensity_filename=self.intensity_filename, sky_box=self.sky_box
        )
        enmap_box = (
            skycoord_box_to_enmap_box(self.sky_box)
            if self.sky_box is not None
            else None
        )
        intensity_shape = enmap.read_map_geometry(str(self.intensity_filename))[0]
        self.intensity = enmap.read_map(
            str(self.intensity_filename),
            sel=0 if len(intensity_shape) > 2 else None,
            box=enmap_box,
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
            box=enmap_box,
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
                str(self.time_filename),
                sel=0 if len(time_shape) > 2 else None,
                box=enmap_box,
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
        self.observation_time = self.observation_start + self.observation_length / 2

        return

    def add_time_offset(self, offset: Time | None):
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
                Time(bunch_read(str(self.info_filename)).t, format="unix")
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
            self.time_first[self.time_first > 0] += offset.unix
        if self.observation_end is None:
            self.observation_end = Time(float(np.amax(self.time_last)), format="unix")

    def _compute_hits(self):
        return (self.inverse_variance > 0).astype(np.int32)

    def _compute_valid_pixel_mask(self):
        if self.finalized:
            bool_map = (abs(self.flux) > 0).astype(np.int32) & (np.isfinite(self.flux))
        else:
            bool_map = (self.inverse_variance > 0).astype(np.int32) & (
                np.isfinite(self.inverse_variance)
            )
        if self.mask is not None:
            bool_map = bool_map & (self.mask > 0)
        return bool_map

    def get_map_id(self):
        return self.__map_id or super().get_map_str_id()

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

    def get_pixel_times(
        self, pix: tuple[int, int]
    ) -> tuple[Time | None, Time | None, Time | None]:
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

    _available_maps: tuple[str, ...] = ("rho", "kappa", "flux", "snr")

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
        self.observation_time = self.observation_start + self.observation_length / 2
        self.sky_box = self.prefiltered_map.sky_box
        self.frequency = self.prefiltered_map.frequency
        self.array = self.prefiltered_map.array
        self.mask = self.prefiltered_map.mask
        self.instrument = self.prefiltered_map.instrument
        if self.prefiltered_map.map_id is not None:
            self.map_id = self.prefiltered_map.map_id
        self._parent_database = self.prefiltered_map._parent_database
        self._hits = self.prefiltered_map._hits
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

    def get_map_id(self):
        return self.__map_id or super().get_map_str_id()

    def add_time_offset(self, offset: Time | None):
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
                Time(bunch_read(str(self.info_filename)).t, format="unix")
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
            self.time_first[self.time_first > 0] += offset.unix
        if self.observation_end is None:
            self.observation_end = Time(float(np.amax(self.time_last)), format="unix")

    def _compute_hits(self):
        return (self.kappa > 0).astype(np.int32)

    def _compute_valid_pixel_mask(self):
        if self.finalized:
            bool_map = (abs(self.flux) > 0).astype(np.int32) & (np.isfinite(self.flux))
        else:
            bool_map = (self.kappa > 0).astype(np.int32) & (np.isfinite(self.kappa))
        if self.mask is not None:
            bool_map = bool_map & (self.mask > 0)
        return bool_map

    def get_snr(self):
        with np.errstate(divide="ignore"):
            snr = self.rho / np.sqrt(self.kappa)

        return snr

    def get_flux(self):
        with np.errstate(divide="ignore"):
            flux = self.rho / self.kappa

        return flux

    def get_pixel_times(
        self, pix: tuple[int, int]
    ) -> tuple[Time | None, Time | None, Time | None]:
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

    sky_box is tuple of SkyCoord: (SkyCoord(ra=ra_min, dec=dec_min), SkyCoord(ra=ra_max, dec=dec_max))
    """

    _available_maps: tuple[str, ...] = ("rho", "kappa", "flux", "snr")

    def __init__(
        self,
        rho_filename: Path,
        kappa_filename: Path,
        start_time: Time,
        end_time: Time,
        sky_box: tuple[SkyCoord, SkyCoord] | None = None,
        time_filename: Path | None = None,
        info_filename: Path | None = None,
        frequency: str | None = None,
        array: str | None = None,
        instrument: str | None = None,
        flux_units: Unit = u.Jy,
        mask: ndmap | None = None,
        map_id: str | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.rho_filename = rho_filename
        self.kappa_filename = kappa_filename
        self.time_filename = time_filename
        self.info_filename = info_filename
        self.observation_start = start_time
        self.observation_end = end_time
        self.sky_box = sky_box
        self.frequency = frequency
        self.array = array
        self.instrument = instrument
        self.flux_units = flux_units
        self.mask = mask
        if map_id is not None:
            self.map_id = map_id
        self._hits = None
        self.log = log or structlog.get_logger()

    @property
    def bbox(self) -> np.ndarray:
        if (box := super().bbox) is not None:
            return box
        shape, wcs = enmap.read_map_geometry(str(self.rho_filename))
        return enmap.box(shape, wcs)

    def build(self):
        log = self.log.bind(rho_filename=self.rho_filename)
        enmap_box = (
            skycoord_box_to_enmap_box(self.sky_box)
            if self.sky_box is not None
            else None
        )
        if enmap_box is not None:
            if enmap_box[0][1] < enmap_box[1][1]:
                enmap_box[1][1] -= 2 * np.pi

        print(str(self.rho_filename))
        if enmap_box is not None:
            if np.isnan(enmap_box).any() or not np.all(np.isfinite(enmap_box)):
                self.log.warning(
                    f"Invalid enmap_box detected for {str(self.rho_filename)}",
                    enmap_box=enmap_box,
                )
                enmap_box = None
        try:
            self.rho = enmap.read_map(str(self.rho_filename), sel=0, box=enmap_box)
            assert type(self.rho) is enmap.ndmap
            print("asserted enmap ndmap")
            print("rho file:", self.rho_filename)
            print("shape:", self.rho.shape)
            print("CRVAL:", self.rho.wcs.wcs.crval)
            print("CRPIX:", self.rho.wcs.wcs.crpix)
            print("CDELT:", self.rho.wcs.wcs.cdelt)
        except (IndexError, AttributeError, AssertionError):
            # Rho map does not have Q, U
            self.rho = enmap.read_map(str(self.rho_filename), box=enmap_box)
            print("rho file:", self.rho_filename)
            print("shape:", self.rho.shape)
            print("CRVAL:", self.rho.wcs.wcs.crval)
            print("CRPIX:", self.rho.wcs.wcs.crpix)
            print("CDELT:", self.rho.wcs.wcs.cdelt)

        log = log.new(kappa_filename=self.kappa_filename)
        try:
            self.kappa = enmap.read_map(str(self.kappa_filename), sel=0, box=enmap_box)
            assert type(self.kappa) is enmap.ndmap
        except (IndexError, AttributeError, AssertionError):
            # Kappa map does not have Q, U
            self.kappa = enmap.read_map(str(self.kappa_filename), box=enmap_box)

        self.map_resolution = u.Quantity(
            abs(self.rho.wcs.wcs.cdelt[0]), self.rho.wcs.wcs.cunit[0]
        )

        log = log.new(time_filename=self.time_filename)
        if self.time_filename is not None:
            # TODO: Handle nuance that the start time is not included.
            try:
                time_map = enmap.read_map(str(self.time_filename), sel=0, box=enmap_box)
                assert type(time_map) is enmap.ndmap
            except (IndexError, AttributeError, AssertionError):
                # if somehow time map requires sel=0
                time_map = enmap.read_map(str(self.time_filename), box=enmap_box)
        else:
            time_map = None
            log.debug("rho_kappa.time.none")

        self.time_first = time_map
        self.time_last = time_map
        self.time_mean = time_map

        self.add_time_offset(self.observation_start)
        self.observation_length = self.observation_end - self.observation_start
        self.observation_time = self.observation_start + self.observation_length / 2
        return

    def add_time_offset(self, offset: Time | None = None):
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
                Time(bunch_read(str(self.info_filename)).t, format="unix")
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
            self.time_first[self.time_first > 0] += offset.unix
        if self.observation_end is None:
            self.observation_end = Time(float(np.amax(self.time_last)), format="unix")

    def get_pixel_times(
        self, pix: tuple[int, int]
    ) -> tuple[Time | None, Time | None, Time | None]:
        return super().get_pixel_times(pix)

    def get_map_id(self):
        return self.__map_id or super().get_map_str_id()

    def _compute_hits(self):
        return (
            (self.kappa > 0).astype(np.int32)
            if not self.finalized
            else (abs(self.flux) > 0).astype(np.int32) & (np.isfinite(self.flux))
        )

    def _compute_valid_pixel_mask(self):
        if self.finalized:
            bool_map = (abs(self.flux) > 0).astype(np.int32) & (np.isfinite(self.flux))
        else:
            bool_map = (self.kappa > 0).astype(np.int32) & (np.isfinite(self.kappa))
        if self.mask is not None:
            bool_map = bool_map & (self.mask > 0)
        return bool_map

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
        #  self.kappa_thresholds = self.get_kappa_thresholds() # can get the kappa thresholds here since it's one per depth1 map (can't do rejecting badmaps here since we're not calling the sources yet, just loading in the maps)
        super().finalize()


class FluxAndSNRMap(ProcessableMap):
    """
    A set of FITS maps read from disk. Could be Depth 1, could
    be monthly or weekly co-adds. Or something else!

    sky_box is tuple of skycoords: (SkyCoord(ra=ra_min, dec=dec_min), SkyCoord(ra=ra_max, dec=dec_max))
    """

    def __init__(
        self,
        flux_filename: Path,
        snr_filename: Path,
        start_time: Time,
        end_time: Time,
        sky_box: tuple[SkyCoord, SkyCoord] | None = None,
        time_filename: Path | None = None,
        info_filename: Path | None = None,
        frequency: str | None = None,
        array: str | None = None,
        instrument: str | None = None,
        flux_units: Unit = u.Jy,
        mask: ndmap | None = None,
        map_id: str | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.flux_filename = flux_filename
        self.snr_filename = snr_filename
        self.time_filename = time_filename
        self.info_filename = info_filename
        self.observation_start = start_time
        self.observation_end = end_time
        self.sky_box = sky_box
        self.frequency = frequency
        self.array = array
        self.instrument = instrument
        self.flux_units = flux_units
        self.mask = mask
        if map_id is not None:
            self.map_id = map_id
        self._hits = None
        self.log = log or structlog.get_logger()

    @property
    def bbox(self) -> np.ndarray:
        if (box := super().bbox) is not None:
            return box
        shape, wcs = enmap.read_map_geometry(str(self.flux_filename))
        return enmap.box(shape, wcs)

    def build(self):
        log = self.log.bind(flux_filename=self.flux_filename)
        enmap_box = (
            skycoord_box_to_enmap_box(self.sky_box)
            if self.sky_box is not None
            else None
        )
        try:
            self.flux = enmap.read_map(str(self.flux_filename), sel=0, box=enmap_box)
            assert type(self.flux) is enmap.ndmap
        except (IndexError, AttributeError, AssertionError):
            # Flux map does not have Q, U
            self.flux = enmap.read_map(str(self.flux_filename), box=enmap_box)

        log = log.new(snr_filename=self.snr_filename)
        try:
            self.snr = enmap.read_map(str(self.snr_filename), sel=0, box=enmap_box)
            assert type(self.snr) is enmap.ndmap
        except (IndexError, AttributeError, AssertionError):
            # SNR map does not have Q, U
            self.snr = enmap.read_map(str(self.snr_filename), box=enmap_box)

        self.map_resolution = u.Quantity(
            abs(self.flux.wcs.wcs.cdelt[0]), self.flux.wcs.wcs.cunit[0]
        )
        if self.flux.shape != self.snr.shape:
            raise ValueError("Flux and SNR maps must have the same shape")
        log = log.new(time_filename=self.time_filename)
        if self.time_filename is not None:
            # TODO: Handle nuance that the start time is not included.
            try:
                time_map = enmap.read_map(str(self.time_filename), sel=0, box=enmap_box)
                assert type(time_map) is enmap.ndmap
            except (IndexError, AttributeError, AssertionError):
                # Somehow time map requires sel=0
                time_map = enmap.read_map(str(self.time_filename), box=enmap_box)
        else:
            time_map = None
            log.debug("flux_snr.time.none")

        self.time_first = time_map
        self.time_last = time_map
        self.time_mean = time_map

        self.add_time_offset(self.observation_start)
        self.observation_length = self.observation_end - self.observation_start
        self.observation_time = self.observation_start + self.observation_length / 2
        return

    def add_time_offset(self, offset: Time | None = None):
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
                Time(bunch_read(str(self.info_filename)).t, format="unix")
                if self.info_filename is not None
                else None
            )
        if offset is None:
            self.log.warning("flux_snr.time_offset.none")
            return
        self.observation_start = offset
        ## TODO: time_first, mean, end are same so this changes all.
        ## this should be changed when we do something smarter.
        if self.time_first is not None:
            self.time_first[self.time_first > 0] += offset.unix
        if self.observation_end is None:
            self.observation_end = Time(float(np.amax(self.time_last)), format="unix")

    def get_pixel_times(
        self, pix: tuple[int, int]
    ) -> tuple[Time | None, Time | None, Time | None]:
        return super().get_pixel_times(pix)

    def get_map_id(self):
        return self.map_id or super().get_map_str_id()

    def _compute_hits(self):
        return (abs(self.flux) > 0).astype(np.int32)

    def _compute_valid_pixel_mask(self):
        bool_map = (abs(self.flux) > 0).astype(np.int32) & (np.isfinite(self.flux))
        if self.mask is not None:
            bool_map = bool_map & (self.mask > 0)
        return bool_map

    def get_snr(self):
        return self.snr

    def get_flux(self):
        return self.flux

    def apply_mask(self):
        return super().apply_mask()

    def finalize(self):
        self.snr = self.get_snr()
        self.flux = self.get_flux()
        self.apply_mask()
        self.finalized = True


class CoaddedRhoKappaMap(ProcessableMap):
    """
    A coadded rho/kappa map created from multiple observations.
    Attributes are the same as RhoAndKappaMap, but the maps are provided
    directly rather than read from disk.

    Coadds are built using map_coadding.MapCoadder classes.
    """

    _available_maps: tuple[str, ...] = ("rho", "kappa", "flux", "snr")

    def __init__(
        self,
        rho: ndmap,
        kappa: ndmap,
        observation_start: Time,
        observation_end: Time,
        time_first: ndmap | None = None,
        time_mean: ndmap | None = None,
        time_last: ndmap | None = None,
        observation_length: TimeDelta | None = None,
        sky_box: tuple[SkyCoord, SkyCoord] | None = None,
        frequency: str | None = None,
        array: str | None = None,
        instrument: str | None = None,
        flux_units: Unit = u.Jy,
        mask: ndmap | None = None,
        map_resolution: u.Quantity | None = None,
        hits: ndmap | None = None,
        map_ids: list = [],
        input_map_times: list = [],
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
        self.observation_time = observation_end - 0.5 * observation_length
        self.sky_box = sky_box
        self.frequency = frequency
        self.array = array
        self.instrument = instrument
        self.flux_units = flux_units
        self.mask = mask
        self._hits = hits
        self.map_resolution = map_resolution
        self.map_ids = map_ids
        self.input_map_times = input_map_times
        self.log = log or structlog.get_logger()

    def build(self):
        pass

    def update_times(self, new_map):
        """
        Update the coadd observation start and observation end
        given the new map.

        Update the time maps themselves as well.
        """
        log = self.log or structlog.get_logger()

        if (
            self.observation_start is None
            or new_map.observation_start < self.observation_start
        ):
            self.observation_start = new_map.observation_start
        if (
            self.observation_end is None
            or new_map.observation_end > self.observation_end
        ):
            self.observation_end = new_map.observation_end
        time_delta = new_map.observation_end - new_map.observation_start
        mid_time = new_map.observation_start + (time_delta / 2)
        self.input_map_times.append(mid_time)

        ## get map union if adding two maps, use hits-weighted mean.
        total_hits = self._compute_hits()
        hit_mask = total_hits > 0
        if isinstance(new_map.time_mean, ndmap):
            if self.time_mean is None:
                self.time_mean = enmap.enmap(new_map.time_mean)
            else:
                self.time_mean = enmap.map_union(
                    self.time_mean * self.hits,
                    new_map.time_mean * new_map.hits,
                )

                self.time_mean[hit_mask] /= total_hits[hit_mask]

        else:
            log.error(
                "CoaddedRhoKappaMap.update_times.no_time_mean",
            )

        if isinstance(new_map.time_first, ndmap):
            if self.time_first is None:
                self.time_first = enmap.enmap(new_map.time_first)
            else:
                self.time_first = pixell_map_union(
                    self.time_first,
                    new_map.time_first,
                    op=np.minimum,
                )
        else:
            log.error(
                "CoaddedRhoKappaMap.update_times.no_time_first",
            )
        if isinstance(new_map.time_last, ndmap):
            if self.time_last is None:
                self.time_last = enmap.enmap(new_map.time_last)
            else:
                self.time_last = pixell_map_union(
                    self.time_last,
                    new_map.time_last,
                    op=np.maximum,
                )
        else:
            log.error(
                "CoaddedRhoKappaMap.update_times.no_time_last",
            )

    def _compute_hits(self):
        return (self.kappa > 0).astype(np.int32)

    def _compute_valid_pixel_mask(self):
        bool_map = (self.kappa > 0).astype(np.int32) & (np.isfinite(self.kappa))
        if self.mask is not None:
            bool_map *= self.mask
        return bool_map

    def get_pixel_times(
        self, pix: tuple[int, int]
    ) -> tuple[Time | None, Time | None, Time | None]:
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
