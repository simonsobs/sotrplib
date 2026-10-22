"""
Dependencies for pre-processing (e.g. map cleaning). Affects
maps in-place. These operations happen in between 'build' and
'finalize' for the map creatino and i/o process.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
import structlog
from astropy import units as u
from astropy.time import Time
from astropydantic import AstroPydanticQuantity
from pixell import enmap
from structlog.types import FilteringBoundLogger

from sotrplib.filters.filters import matched_filter_depth1_map
from sotrplib.maps.core import (
    MatchedFilteredIntensityAndInverseVarianceMap,
    ProcessableMap,
)

# avoid importing maps at module import time to prevent circular imports;
# import inside methods where needed
from sotrplib.maps.masks import (
    mask_asteroids,
    mask_asteroids_socat,
    mask_dustgal,
    mask_edge,
    mask_planets,
)
from sotrplib.solar_system.solar_system import load_jpl_ephem_database
from sotrplib.utils.utils import get_frequency, get_fwhm


class MapPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        """
        Provides preprocessing for the input map
        """
        return input_map


class PlanetMasker(MapPreprocessor):
    mask_radius: AstroPydanticQuantity = 15 * u.arcmin

    def __init__(
        self,
        mask_radius: AstroPydanticQuantity = 15 * u.arcmin,
        log: FilteringBoundLogger | None = None,
    ):
        self.mask_radius = mask_radius
        self.log = log or structlog.get_logger()

    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        log = self.log.bind(func="PlanetMasker.preprocess")
        log.info(
            "PlanetMasker.preprocess.masking_planets",
            mask_radius=self.mask_radius,
            time_range=[input_map.observation_start, input_map.observation_end],
        )
        planet_mask = mask_planets(
            input_map.time_mean,
            [input_map.observation_start, input_map.observation_end],
            mask_radius=self.mask_radius,
            log=log,
        )
        input_map.mask = (
            input_map.mask * planet_mask if input_map.mask is not None else planet_mask
        )
        log.info("PlanetMasker.preprocess.completed")
        return input_map


class AsteroidMasker(MapPreprocessor):
    """
    Mask the solar-system objects that cross each map in its observation
    window.

    - First method: SOCat (source_type "sso"). SOCat uses the per-pixel time
      of the map to refine each position.
    - Second method: a local ephemeris file (see
      sotrplib.solar_system.download_ephem_from_horizons). The masker uses
      it only if SOCat is not configured or a query fails. If SOCat finds
      no objects, the masker does not use the file.

    The masker loads the ephemeris file one time, not for each map.
    """

    mask_radius: AstroPydanticQuantity = 10 * u.arcmin

    def __init__(
        self,
        use_socat: bool = True,
        ephem_file_path: Path | None = None,
        start_time: Time | None = None,
        end_time: Time | None = None,
        mask_radius: AstroPydanticQuantity = 10 * u.arcmin,
        interp_time_range: AstroPydanticQuantity = 0.5 * u.day,
        interp_to: AstroPydanticQuantity | None = 10 * u.min,
        log: FilteringBoundLogger | None = None,
    ):
        self.mask_radius = mask_radius
        self.interp_time_range = interp_time_range
        self.interp_to = interp_to
        self.log = log or structlog.get_logger()

        self.socat = None
        if use_socat:
            try:
                from sotrplib.source_catalog.socat import SOCat

                self.socat = SOCat(log=self.log)
            except Exception:
                self.log.warning("AsteroidMasker.init.socat_unavailable", exc_info=True)

        self.ephem_df = None
        if ephem_file_path is not None:
            self.ephem_df = load_jpl_ephem_database(
                ephem_file_path=ephem_file_path,
                start_time=start_time,
                stop_time=end_time,
                log=self.log,
            )

        if self.socat is None and self.ephem_df is None:
            self.log.error(
                "AsteroidMasker.init.no_method_available",
                message="Neither SOCat nor a local ephem_file_path is available -- "
                "asteroid masking will be a no-op.",
            )

    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        log = self.log.bind(func="AsteroidMasker.preprocess")
        log.info(
            "AsteroidMasker.preprocess.masking_asteroids",
            mask_radius=self.mask_radius,
            time_range=[input_map.observation_start, input_map.observation_end],
        )

        asteroid_mask = None
        if self.socat is not None:
            try:
                asteroid_mask = mask_asteroids_socat(
                    input_map=input_map,
                    socat=self.socat,
                    mask_radius=self.mask_radius,
                    log=log,
                )
            except Exception:
                log.warning(
                    "AsteroidMasker.preprocess.socat_query_failed", exc_info=True
                )
                asteroid_mask = None

        if asteroid_mask is None and self.ephem_df is not None:
            log.info("AsteroidMasker.preprocess.using_local_ephem_fallback")
            asteroid_mask = mask_asteroids(
                input_map=input_map,
                ephem_df=self.ephem_df,
                mask_radius=self.mask_radius,
                interp_time_range=self.interp_time_range,
                interp_to=self.interp_to,
                log=log,
            )

        if asteroid_mask is None:
            log.error("AsteroidMasker.preprocess.no_method_available")
            return input_map

        input_map.mask = (
            input_map.mask * asteroid_mask
            if input_map.mask is not None
            else asteroid_mask
        )
        log.info("AsteroidMasker.preprocess.completed")
        return input_map


class KappaRhoCleaner(MapPreprocessor):
    cut_on: Literal["median", "percentile", "max"] = "median"
    fraction: float = 0.05

    def __init__(
        self,
        cut_on: Literal["median", "percentile", "max"] = "median",
        fraction: float = 0.05,
    ):
        self.cut_on = cut_on
        self.fraction = fraction

    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        # TODO: Figure out what warnings this raises
        from sotrplib.maps.maps import clean_map, kappa_clean

        input_map.kappa = kappa_clean(kappa=input_map.kappa, rho=input_map.rho)
        input_map.rho = clean_map(
            imap=input_map.rho,
            inverse_variance=input_map.kappa,
            cut_on=self.cut_on,
            fraction=self.fraction,
        )

        return input_map


class MatchedFilter(MapPreprocessor):
    def __init__(
        self,
        infofile: Path | None = None,
        maskfile: Path | None = None,
        beam1d: Path | dict[str, Path] | None = None,
        shrink_holes: AstroPydanticQuantity = 20 * u.arcmin,
        apod_edge: AstroPydanticQuantity = 10 * u.arcmin,
        apod_holes: AstroPydanticQuantity = 5 * u.arcmin,
        noisemask_lim: float | None = None,
        noisemask_radius: AstroPydanticQuantity = 10 * u.arcmin,
        highpass: bool = False,
        band_height: AstroPydanticQuantity = 1 * u.degree,
        shift: float = 0,
        simple: bool = False,
        simple_lknee: float = 1000,
        simple_alpha: float = -3.5,
        lres=(70, 100),
        pixwin: str = "nn",
        noise_mask_max_rms: float | None = None,
        noise_mask_tile_size: AstroPydanticQuantity = 0.5 * u.degree,
        noise_mask_grow_tiles: int = 1,
        noise_mask_apod: AstroPydanticQuantity = 10 * u.arcmin,
        noise_mask_max_fraction: float = 0.5,
        log: FilteringBoundLogger | None = None,
    ):
        """
        Matched filter of a depth-1 map (see matched_filter_depth1_map).

        noise_mask_max_rms: if not None, filter the map two times. The first
            filter gives the snr map. Tiles of noise_mask_tile_size where the
            robust rms of the snr is more than noise_mask_max_rms have much
            more noise than the noise model (e.g. scan stripes). The second
            filter removes these tiles from the noise model, so that the noise
            model is correct for the remaining map.
        noise_mask_grow_tiles: also remove this number of tiles around each
            noisy tile.
        noise_mask_apod: apodization width of the edge of the removed region.
        noise_mask_max_fraction: if the noisy tiles are more than this fraction
            of the map, the second filter does not occur, because the noise
            model needs sufficient area.
        """
        self.infofile = infofile
        self.maskfile = maskfile
        self.beam1d = beam1d
        self.shrink_holes = shrink_holes
        self.apod_edge = apod_edge
        self.apod_holes = apod_holes
        self.noisemask_lim = noisemask_lim
        self.noisemask_radius = noisemask_radius
        self.highpass = highpass
        self.band_height = band_height
        self.shift = shift
        self.simple = simple
        self.simple_lknee = simple_lknee
        self.simple_alpha = simple_alpha
        self.lres = lres
        self.pixwin = pixwin
        self.noise_mask_max_rms = noise_mask_max_rms
        self.noise_mask_tile_size = noise_mask_tile_size
        self.noise_mask_grow_tiles = noise_mask_grow_tiles
        self.noise_mask_apod = noise_mask_apod
        self.noise_mask_max_fraction = noise_mask_max_fraction
        self.log = log

    def _beam1d_for(self, input_map: ProcessableMap) -> Path | None:
        if not isinstance(self.beam1d, dict):
            return self.beam1d
        try:
            return self.beam1d[input_map.frequency]
        except KeyError:
            raise ValueError(
                f"No beam1d profile for band '{input_map.frequency}' of map "
                f"{input_map.map_name}; beam1d has bands {sorted(self.beam1d)}."
            ) from None

    def _filter(
        self, input_map: ProcessableMap, noise_model_mask: np.ndarray | None = None
    ) -> tuple[enmap.ndmap, enmap.ndmap]:
        return matched_filter_depth1_map(
            imap=input_map.intensity * input_map.intensity_units.to(u.K),
            ivarmap=input_map.inverse_variance / input_map.intensity_units.to(u.K) ** 2,
            band_center=get_frequency(
                freq=input_map.frequency,
                arr=input_map.array,
                instrument=input_map.instrument,
            ),
            infofile=self.infofile,
            maskfile=self.maskfile,
            source_mask=input_map.mask,
            beam_fwhm=get_fwhm(
                freq=input_map.frequency,
                arr=input_map.array,
                instrument=input_map.instrument,
            ),
            beam1d=self._beam1d_for(input_map),
            shrink_holes=self.shrink_holes,
            apod_edge=self.apod_edge,
            apod_holes=self.apod_holes,
            noisemask_lim=self.noisemask_lim,
            noisemask_radius=self.noisemask_radius,
            highpass=self.highpass,
            band_height=self.band_height,
            shift=self.shift,
            simple=self.simple,
            simple_lknee=self.simple_lknee,
            simple_alpha=self.simple_alpha,
            lres=self.lres,
            pixwin=self.pixwin,
            noise_model_mask=noise_model_mask,
        )

    def _noise_model_mask(
        self, input_map: ProcessableMap, rho: enmap.ndmap, kappa: enmap.ndmap
    ) -> np.ndarray | None:
        """
        Make the noise model mask (0 in the noisy tiles, apodized) from the snr
        of the first filter. Returns None if no tile is noisy, or if the noisy
        tiles are more than noise_mask_max_fraction of the map.
        """
        from sotrplib.maps.maps import get_snr, snr_rms_tile_mask

        log = (self.log or structlog.get_logger()).bind(
            func_name="matched_filter.noise_model_mask", map_name=input_map.map_name
        )
        kappa_values = np.asarray(kappa)
        valid = kappa_values > 0
        if input_map.mask is not None:
            valid &= np.asarray(input_map.mask) > 0
        snr = get_snr(np.asarray(rho), kappa_values)
        resolution = u.Quantity(abs(rho.wcs.wcs.cdelt[1]), rho.wcs.wcs.cunit[1])
        tile_pix = max(
            1, int(round((self.noise_mask_tile_size / resolution).to_value("")))
        )
        noisy, rms = snr_rms_tile_mask(
            snr,
            valid,
            tile_pix,
            max_rms=self.noise_mask_max_rms,
            grow_tiles=self.noise_mask_grow_tiles,
        )
        fraction = float(noisy[valid].mean()) if valid.any() else 0.0
        log = log.bind(
            tile_rms_median=float(np.nanmedian(rms))
            if np.isfinite(rms).any()
            else None,
            noisy_fraction=round(fraction, 4),
        )
        if not noisy.any():
            log.info("matched_filter.noise_model_mask.no_noisy_tiles")
            return None
        if fraction > self.noise_mask_max_fraction:
            log.warning(
                "matched_filter.noise_model_mask.too_many_noisy_tiles",
                max_fraction=self.noise_mask_max_fraction,
            )
            return None
        log.info("matched_filter.noise_model_mask.applied")
        keep = enmap.ndmap((~noisy).astype(np.float32), rho.wcs)
        return np.asarray(
            enmap.apod_mask(keep > 0, self.noise_mask_apod.to_value(u.rad))
        )

    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        rho, kappa = self._filter(input_map)
        if self.noise_mask_max_rms is not None:
            noise_model_mask = self._noise_model_mask(input_map, rho, kappa)
            if noise_model_mask is not None:
                del rho, kappa
                rho, kappa = self._filter(input_map, noise_model_mask=noise_model_mask)
        return MatchedFilteredIntensityAndInverseVarianceMap(
            rho=rho, kappa=kappa, prefiltered_map=input_map, flux_units=u.mJy
        )


class EdgeMask(MapPreprocessor):
    edge_width: AstroPydanticQuantity = AstroPydanticQuantity(10.0 * u.arcmin)
    mask_on: Literal["rho", "kappa", "flux", "snr", "inverse_variance", "intensity"] = (
        "kappa"
    )

    def __init__(
        self,
        edge_width: u.Quantity = u.Quantity(10.0, "arcmin"),
        mask_on: Literal[
            "rho", "kappa", "flux", "snr", "inverse_variance", "intensity"
        ] = "kappa",
        log: FilteringBoundLogger | None = None,
    ):
        self.edge_width = edge_width
        self.mask_on = mask_on
        self.log = log or structlog.get_logger()

    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        self.log = self.log.bind(func="EdgeMask.preprocess")
        number_of_pixels = int(
            self.edge_width.to_value(u.arcmin)
            / input_map.map_resolution.to_value(u.arcmin)
        )
        self.log = self.log.bind(number_of_pixels=number_of_pixels)
        mask_map = (
            getattr(input_map, self.mask_on)
            if hasattr(input_map, self.mask_on)
            else None
        )
        if mask_map is None:
            self.log.error(
                "EdgeMask.preprocess.get_mask_map.no_attribute", mask_on=self.mask_on
            )
            return input_map

        edge_mask = mask_edge(imap=mask_map, pix_num=number_of_pixels)
        input_map.mask = (
            edge_mask if input_map.mask is None else input_map.mask * edge_mask
        )
        self.log.info("EdgeMask.preprocess completed")

        return input_map


class GalaxyMask(MapPreprocessor):
    mask_path: Path

    def __init__(
        self,
        mask_path: Path | None = None,
        invert: bool = False,
        mask_map: enmap.ndmap | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.mask_path = mask_path
        self.mask_map = mask_map
        self.invert = invert
        self.log = log or structlog.get_logger()

    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        log = self.log.bind(func="GalaxyMask.preprocess")
        if self.mask_map is None and self.mask_path is None:
            log.error(
                "GalaxyMask.preprocess.no_mask",
                message="No mask provided for GalaxyMask preprocessor",
            )
            return input_map

        if self.mask_map is None and self.mask_path is not None:
            self.mask_map = enmap.read_map(str(self.mask_path))

        galaxy_mask = mask_dustgal(
            input_map.hits,
            galmask=self.mask_map,
            log=log,
        )
        if self.invert:
            galaxy_mask = 1 - galaxy_mask
        input_map.mask = (
            input_map.mask * galaxy_mask if input_map.mask is not None else galaxy_mask
        )

        log.info("GalaxyMask.preprocess.completed")
        return input_map
