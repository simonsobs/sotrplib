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
from mapcat.pointing.const import ConstantPointingModel
from pixell import enmap
from pixell.enmap import ndmap
from pydantic import AwareDatetime
from structlog.types import FilteringBoundLogger

from sotrplib.maps.utils import pixell_map_union, skycoord_box_to_enmap_box

PointingModel = ConstantPointingModel


class ProcessableMap(ABC):
    """Abstract base class for pipeline-processable sky maps.

    All maps expose two primary products after ``finalize`` is called:

    - ``snr`` — the signal-to-noise ratio map
    - ``flux`` — the calibrated flux map

    Maps are created in two steps.  ``build`` reads or simulates the raw map
    data, and ``finalize`` converts the internal representation (e.g.
    rho/kappa or intensity/ivar) into the SNR and flux arrays required by the
    pipeline.

    The raw ``rho`` and ``kappa`` attributes are settable before ``finalize``
    is called.  Once ``finalize`` is called they are deleted from memory and
    can no longer be updated, because the derived flux map would otherwise
    become inconsistent.
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

    sky_box: tuple[SkyCoord, SkyCoord] | None = None
    """
    Sky bounding box. Can be set to initialize the map with a cutout of a larger map.
    Should be in the form

    ``sky_box = (SkyCoord(ra=ra_min, dec=dec_min), SkyCoord(ra=ra_max, dec=dec_max))``

    Wrap detection (both RAs in ``[0, 2π)``):

    - ``sky_box[0].ra < sky_box[1].ra`` — non-wrapping region.
    - ``sky_box[0].ra > sky_box[1].ra`` — wrapping region (spans RA = 0);
      ``sky_box[0].ra`` is the upper boundary (near 2π) and ``sky_box[1].ra`` is the
      lower boundary (near 0).
    """

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
        """Read or simulate the raw map data.

        Populates internal attributes (e.g. ``rho``/``kappa`` or
        ``intensity``/``inverse_variance``) and the time maps.  Must be called
        before ``finalize``.
        """
        return

    @abstractmethod
    def _compute_hits(self):
        """Compute and return the hits map.

        Returns
        -------
        enmap.ndmap
            Integer map counting the number of times each pixel was observed.
        """
        raise NotImplementedError(
            "_compute_hits must be implemented by ProcessableMap subclass"
        )

    def _core_filter_sources(self, source_positions: SkyCoord, bool_map: enmap.ndmap):
        """Return the subset of sources that fall in valid (unmasked) pixels.

        Parameters
        ----------
        source_positions : SkyCoord
            Candidate source positions to filter.
        bool_map : enmap.ndmap
            Boolean validity map: 1/True in valid pixels, 0/False in masked or
            unobserved pixels.

        Returns
        -------
        mask : np.ndarray of bool
            Per-source boolean indicating which sources pass the filter.
        filtered_positions : SkyCoord
            Source positions that fall in valid pixels.
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
        """Return the subset of sources that lie in valid, unmasked pixels.

        Parameters
        ----------
        source_positions : SkyCoord
            Candidate source positions.

        Returns
        -------
        mask : np.ndarray of bool
            Per-source boolean mask.
        filtered_positions : SkyCoord
            Positions of sources in valid pixels.
        """
        return self._core_filter_sources(source_positions, self.get_valid_pixel_mask())

    @property
    def hits(self):
        """Hits map: number of times each pixel was observed.

        Computed lazily on first access and cached.

        Returns
        -------
        enmap.ndmap
            Integer hits map.
        """
        if self._hits is None:
            self._hits = self._compute_hits()
        return self._hits

    @property
    def noise(self):
        """Noise map derived as ``flux / snr``.

        Returns
        -------
        enmap.ndmap
            Per-pixel noise estimate.
        """
        with np.errstate(divide="ignore"):
            return self.flux / self.snr

    @property
    def rho(self):
        """Matched-filter numerator map, computed from ``snr`` and ``flux``.

        Returns ``snr² / flux`` unless a rho map was set explicitly.

        Raises
        ------
        AttributeError
            If called after ``finalize``.

        Returns
        -------
        enmap.ndmap
            Rho map.
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
        """Matched-filter denominator map, computed from ``snr`` and ``flux``.

        Returns ``snr² / flux²`` unless a kappa map was set explicitly.

        Raises
        ------
        AttributeError
            If called after ``finalize``.

        Returns
        -------
        enmap.ndmap
            Kappa map.
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
        """Bounding box of the map in the pixell enmap format.

        Searches ``_available_maps`` in order for the first loaded map and
        returns ``enmap.box(shape, wcs)``.  Falls back to
        ``skycoord_box_to_enmap_box(self.sky_box)`` when ``sky_box`` is set.

        Returns
        -------
        np.ndarray or None
            Bounding box array of shape ``(2, 2)`` in radians, or ``None`` if
            neither a map nor a ``sky_box`` is available.
        """
        for attr in self._available_maps:
            if (m := getattr(self, attr, None)) is not None:
                return enmap.box(m.shape, m.wcs)
        if self.sky_box is not None:
            return skycoord_box_to_enmap_box(self.sky_box)
        return None

    @property
    def map_id(self) -> str:
        """Unique string identifier for this map.

        Returns the explicitly set ID if one has been provided, otherwise
        falls back to ``"{frequency}_{array}_{observation_start_timestamp}"``.

        Returns
        -------
        str
            Map identifier.
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
        """Build a human-readable string identifier from map metadata.

        Returns
        -------
        str
            String of the form ``"{frequency}_{array}_{observation_start_unix}"``.
        """
        return (
            f"{self.frequency}_{self.array}_{int(self.observation_start.timestamp())}"
        )

    def get_pixel_times(self, pix: tuple[int, int]):
        """Return the observation start, mean, and end time for a given pixel.

        Parameters
        ----------
        pix : tuple of int
            Pixel coordinates ``(dec_pix, ra_pix)``.

        Returns
        -------
        t_start : datetime or ndmap value
            Time of the first observation of the pixel.
        t_mean : datetime or ndmap value
            Mean observation time of the pixel.
        t_end : datetime or ndmap value
            Time of the last observation of the pixel.
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
            else (self.observation_end - self.observation_start) / 2
            + self.observation_start
        )
        t_end = (
            self.time_last[x, y] if self.time_last is not None else self.observation_end
        )
        return t_start, t_mean, t_end

    def apply_mask(self):
        """Multiply the SNR and flux maps by the mask in place.

        Has no effect if ``self.mask`` is ``None``.  Masked pixels are
        indicated by 0 in the mask map.
        """
        if self.mask is not None:
            self.snr *= self.mask
            self.flux *= self.mask
        return

    def finalize(self):
        """Convert the raw map data into SNR and flux arrays.

        Called after ``build`` and any preprocessing steps.  Clears the cached
        valid-pixel mask (which may be invalidated by masking), applies
        ``self.mask`` to the SNR and flux maps, and sets ``self.finalized =
        True``.  After this call ``rho`` and ``kappa`` are no longer accessible.
        """
        del self.rho
        del self.kappa
        self._valid_pixel_mask_cache = None
        self.apply_mask()
        self.finalized = True

        return


class IntensityAndInverseVarianceMap(ProcessableMap):
    """A processable map backed by intensity and inverse-variance FITS files.

    Suitable for Depth-1 maps or co-adds stored as separate intensity and
    inverse-variance FITS files.  Call ``build`` to read the files, then
    ``finalize`` to compute ``snr`` and ``flux``.

    Parameters
    ----------
    intensity_filename : Path
        Path to the intensity FITS file.
    inverse_variance_filename : Path
        Path to the inverse-variance FITS file.
    start_time : AwareDatetime
        Observation start time.
    end_time : AwareDatetime
        Observation end time.
    sky_box : tuple of SkyCoord, optional
        Bounding box for a cutout read.
    time_filename : Path, optional
        Path to the per-pixel time FITS file.
    info_filename : Path, optional
        Path to the observation info HDF file (used for the time offset).
    frequency : str, optional
        Frequency band string (e.g. ``"f090"``).
    array : str, optional
        Detector array/wafer name.
    instrument : str, optional
        Instrument name (e.g. ``"SOLAT"``).
    matched_filtered : bool, optional
        If ``True``, treat intensity as rho and ivar as kappa when computing
        the flux.  Default is ``False``.
    mask : ndmap, optional
        Pre-computed mask (1 = valid, 0 = masked).
    intensity_units : Unit, optional
        Physical units of the intensity map.  Default is Kelvin.
    map_id : str, optional
        Override the auto-generated map identifier.
    log : FilteringBoundLogger, optional
        Structured logger.
    """

    _available_maps: tuple[str, ...] = ("intensity",)

    def __init__(
        self,
        intensity_filename: Path,
        inverse_variance_filename: Path,
        start_time: AwareDatetime,
        end_time: AwareDatetime,
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

        return

    def add_time_offset(self, offset: timedelta):
        """Add an absolute time offset to the per-pixel time maps.

        ACT time maps store seconds elapsed since the start of the observation.
        This method converts them to absolute unix timestamps by adding the
        observation start time.

        Parameters
        ----------
        offset : datetime or timedelta
            Absolute time to add.  If ``None``, the offset is read from the
            info HDF file.  If neither is available, the method is a no-op.
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

    def get_pixel_times(self, pix: tuple[int, int]):
        return super().get_pixel_times(pix)

    def apply_mask(self):
        return super().apply_mask()

    def finalize(self):
        self.snr = self.get_snr()
        self.flux = self.get_flux()
        super().finalize()


class MatchedFilteredIntensityAndInverseVarianceMap(ProcessableMap):
    """A processable map containing matched-filtered rho/kappa arrays.

    Created by the ``MatchedFilter`` preprocessor from an
    ``IntensityAndInverseVarianceMap``.  Inherits observation metadata from
    the source map.

    Parameters
    ----------
    rho : enmap.ndmap
        Matched-filter numerator map.
    kappa : enmap.ndmap
        Matched-filter denominator map.
    flux_units : astropy.units.Unit
        Physical units of the derived flux map.
    prefiltered_map : IntensityAndInverseVarianceMap
        The source map from which observation metadata is inherited.
    keep_prefiltered : bool, optional
        If ``True``, retain the original intensity and ivar arrays as
        ``prefiltered_intensity`` and ``prefiltered_inverse_variance``.
        Default is ``False``.
    log : FilteringBoundLogger, optional
        Structured logger.
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

    def get_pixel_times(self, pix: tuple[int, int]):
        return super().get_pixel_times(pix)

    def apply_mask(self):
        return super().apply_mask()

    def finalize(self):
        self.snr = self.get_snr()
        self.flux = self.get_flux()
        super().finalize()


class RhoAndKappaMap(ProcessableMap):
    """A processable map backed by rho and kappa FITS files on disk.

    Suitable for Depth-1 maps or co-adds stored as matched-filtered rho/kappa
    pairs.  Call ``build`` to read the files, then ``finalize`` to compute
    ``snr`` and ``flux``.

    Parameters
    ----------
    rho_filename : Path
        Path to the rho FITS file.
    kappa_filename : Path
        Path to the kappa FITS file.
    start_time : AwareDatetime
        Observation start time.
    end_time : AwareDatetime
        Observation end time.
    sky_box : tuple of SkyCoord, optional
        Bounding box ``(SkyCoord(ra=ra_min, dec=dec_min), SkyCoord(ra=ra_max,
        dec=dec_max))`` for a cutout read.
    time_filename : Path, optional
        Path to the per-pixel time FITS file.
    info_filename : Path, optional
        Path to the observation info HDF file.
    frequency : str, optional
        Frequency band string (e.g. ``"f090"``).
    array : str, optional
        Detector array/wafer name.
    instrument : str, optional
        Instrument name (e.g. ``"SOLAT"``).
    flux_units : Unit, optional
        Physical units of the derived flux map.  Default is Jy.
    mask : ndmap, optional
        Pre-computed mask (1 = valid, 0 = masked).
    map_id : str, optional
        Override the auto-generated map identifier.
    log : FilteringBoundLogger, optional
        Structured logger.
    """

    _available_maps: tuple[str, ...] = ("rho", "kappa", "flux", "snr")

    def __init__(
        self,
        rho_filename: Path,
        kappa_filename: Path,
        start_time: AwareDatetime,
        end_time: AwareDatetime,
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
        try:
            self.rho = enmap.read_map(str(self.rho_filename), sel=0, box=enmap_box)
            assert type(self.rho) is enmap.ndmap
        except (IndexError, AttributeError, AssertionError):
            # Rho map does not have Q, U
            self.rho = enmap.read_map(str(self.rho_filename), box=enmap_box)

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

    def get_map_id(self):
        return self.__map_id or super().get_map_str_id()

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

    def apply_mask(self):
        return super().apply_mask()

    def finalize(self):
        self.snr = self.get_snr()
        self.flux = self.get_flux()
        super().finalize()


class FluxAndSNRMap(ProcessableMap):
    """A processable map backed by pre-computed flux and SNR FITS files.

    Use this when the flux and SNR maps have already been computed and written
    to disk (e.g. by an upstream pipeline step).

    Parameters
    ----------
    flux_filename : Path
        Path to the flux FITS file.
    snr_filename : Path
        Path to the SNR FITS file.
    start_time : AwareDatetime
        Observation start time.
    end_time : AwareDatetime
        Observation end time.
    sky_box : tuple of SkyCoord, optional
        Bounding box ``(SkyCoord(ra=ra_min, dec=dec_min), SkyCoord(ra=ra_max,
        dec=dec_max))`` for a cutout read.
    time_filename : Path, optional
        Path to the per-pixel time FITS file.
    info_filename : Path, optional
        Path to the observation info HDF file.
    frequency : str, optional
        Frequency band string (e.g. ``"f090"``).
    array : str, optional
        Detector array/wafer name.
    instrument : str, optional
        Instrument name (e.g. ``"SOLAT"``).
    flux_units : Unit, optional
        Physical units of the flux map.  Default is Jy.
    mask : ndmap, optional
        Pre-computed mask (1 = valid, 0 = masked).
    map_id : str, optional
        Override the auto-generated map identifier.
    log : FilteringBoundLogger, optional
        Structured logger.
    """

    def __init__(
        self,
        flux_filename: Path,
        snr_filename: Path,
        start_time: AwareDatetime,
        end_time: AwareDatetime,
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
            self.log.warning("flux_snr.time_offset.none")
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
    """A coadded rho/kappa map assembled in memory from multiple observations.

    Unlike ``RhoAndKappaMap``, the arrays are provided directly rather than
    read from disk.  Instances are normally created by
    ``map_coadding.RhoKappaMapCoadder``.

    Parameters
    ----------
    rho : ndmap
        Coadded rho map.
    kappa : ndmap
        Coadded kappa map.
    observation_start : AwareDatetime
        Start time of the earliest contributing observation.
    observation_end : AwareDatetime
        End time of the latest contributing observation.
    time_first : ndmap, optional
        Per-pixel first-observation time map.
    time_mean : ndmap, optional
        Per-pixel mean-observation time map.
    time_last : ndmap, optional
        Per-pixel last-observation time map.
    observation_length : timedelta, optional
        Total span of the coadd.
    sky_box : tuple of SkyCoord, optional
        Bounding box of the coadd.
    frequency : str, optional
        Frequency band string.
    array : str, optional
        Array/wafer name (or ``"coadd"`` for multi-array coadds).
    instrument : str, optional
        Instrument name.
    flux_units : Unit, optional
        Physical units of the derived flux.  Default is Jy.
    mask : ndmap, optional
        Combined mask for the coadd.
    map_resolution : Quantity, optional
        Pixel resolution.
    hits : ndmap, optional
        Pre-computed hits map.
    map_ids : list, optional
        List of map IDs that were coadded.
    input_map_times : list, optional
        Mid-point observation times of the individual maps.
    log : FilteringBoundLogger, optional
        Structured logger.
    """

    _available_maps: tuple[str, ...] = ("rho", "kappa", "flux", "snr")

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
        """Extend the coadd time bounds and per-pixel time maps to include a new map.

        Parameters
        ----------
        new_map : ProcessableMap
            Map being added to the coadd.  Its observation times and per-pixel
            time maps are merged into the coadd using min/max/hits-weighted-mean
            operations.
        """
        log = self.log or structlog.get_logger()

        self.observation_start = (
            new_map.observation_start
            if self.observation_start is None
            else min(self.observation_start, new_map.observation_start)
        )
        self.observation_end = (
            new_map.observation_end
            if self.observation_end is None
            else max(self.observation_end, new_map.observation_end)
        )
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
