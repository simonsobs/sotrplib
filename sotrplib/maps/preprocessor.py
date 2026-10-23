"""
Dependencies for pre-processing (e.g. map cleaning). Affects
maps in-place. These operations happen in between 'build' and
'finalize' for the map creatino and i/o process.
"""

import tempfile
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
from sotrplib.maps import gfal
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
        log: FilteringBoundLogger | None = None,
    ):
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

    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        rho, kappa = matched_filter_depth1_map(
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
        )
        return MatchedFilteredIntensityAndInverseVarianceMap(
            rho=rho, kappa=kappa, prefiltered_map=input_map, flux_units=u.mJy
        )


class SPT3GTransientFilter(MapPreprocessor):
    """
    Matched-filters an SPT3G map using spt3g_software's own native transient
    pipeline (``spt3g.transients.TransientMapFiltering``), instead of the
    generic ACT/tenki-derived :func:`matched_filter_depth1_map` used by
    :class:`MatchedFilter`.

    Mirrors ``transients/scripts/filter_maps.py`` (spt3g_software's own CLI
    for this) as closely as possible for the no-coadd case: the file is fed
    through a real ``core.G3Pipeline`` via ``core.G3Reader`` (not just the
    extracted Map frame), followed by ``maps.map_modules.MakeMapsUnpolarized``
    and then ``TransientMapFiltering`` with ``subtract_coadd=False`` (no
    per-field coadd map is staged in sotrplib's map reading) but
    ``normalize_weights=True`` and ``compute_noisechisq=True`` -- those are
    only disabled by the script's ``--filter-coadd`` mode, not by the absence
    of a coadd. Critically, the map is left **weighted** going into the
    pipeline, exactly as the script does without a coadd:
    ``SubtractReferenceMap`` (which unweights) only runs when a coadd is
    given, and ``ConvertMapUnitsToJansky`` has its own branch to handle an
    already-weighted map correctly. Explicitly unweighting first (an earlier
    version of this class did) puts the map in a state the rest of the
    pipeline doesn't expect and was the likely cause of visibly under-filtered
    (speckled, not beam-correlated) SNR maps.

    ``TransientMapFiltering`` operates on native ``G3SkyMap``/
    ``G3SkyMapWeights`` frame objects, not the enmap intensity/
    inverse_variance arrays already extracted onto a
    ``SPTIntensityAndInverseVarianceMap``, so this preprocessor re-reads
    ``input_map``'s original ``.g3`` file (gfal-copying it to scratch again
    if it's a grid URL) rather than operating on ``input_map.intensity``/
    ``input_map.inverse_variance`` directly. ``input_map`` must therefore
    expose ``map_filename`` and ``scratch_dir`` (as
    ``SPTIntensityAndInverseVarianceMap`` does).

    Band-specific flux calibration (K -> mJy), the high-pass radial filter,
    and the Gaussian matched-filter FWHM are spt3g_software's own hardcoded
    per-band values (see ``transients/python/filter_tools.py``), not
    sotrplib's ``utils.get_fwhm``/``get_frequency`` lookup used by
    :class:`MatchedFilter`. Source masking (``MaskMaps``) is off by default,
    matching the script's ``--mask``/``--field`` (masking requires a field);
    it's otherwise handled separately by sotrplib's own mask preprocessors.

    The output is a weighted-T / inverse-variance pair in mJy, exactly the
    rho/kappa convention :class:`MatchedFilteredIntensityAndInverseVarianceMap`
    already uses (flux = rho / kappa, snr = rho / sqrt(kappa)), so it slots
    into the rest of the pipeline identically to :class:`MatchedFilter`'s
    output.
    """

    def __init__(
        self,
        mask: bool = False,
        field: str | None = None,
        high_pass: bool = True,
        match_sources: bool = True,
        normalize_weights: bool = True,
        compute_noisechisq: bool = True,
        compute_snr_annulus: bool = False,
        log: FilteringBoundLogger | None = None,
    ):
        if mask and field is None:
            raise ValueError("field is required when mask=True.")
        self.mask = mask
        self.field = field
        self.high_pass = high_pass
        self.match_sources = match_sources
        self.normalize_weights = normalize_weights
        self.compute_noisechisq = compute_noisechisq
        self.compute_snr_annulus = compute_snr_annulus
        self.log = log or structlog.get_logger()

    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        if not hasattr(input_map, "map_filename"):
            raise TypeError(
                f"{type(input_map).__name__} has no map_filename; "
                "SPT3GTransientFilter requires a map built from a raw .g3 "
                "file (e.g. SPTIntensityAndInverseVarianceMap)."
            )

        from spt3g import core, transients
        from spt3g import maps as spt3g_maps

        log = self.log.bind(
            func="SPT3GTransientFilter.preprocess",
            map_filename=str(input_map.map_filename),
        )
        band = input_map.frequency
        scratch_dir = (
            getattr(input_map, "scratch_dir", None)
            or Path(tempfile.gettempdir()) / "sotrplib_spt"
        )

        results = []

        def _collect(frame):
            if frame.type == core.G3FrameType.Map:
                results.append(frame)

        with gfal.fetch_to_scratch(
            str(input_map.map_filename), scratch_dir, log=log
        ) as local_g3:
            pipe = core.G3Pipeline()
            pipe.Add(core.G3Reader, filename=str(local_g3))
            pipe.Add(spt3g_maps.map_modules.MakeMapsUnpolarized)
            pipe.Add(
                transients.TransientMapFiltering,
                bands=[band],
                field=self.field,
                mask=self.mask,
                subtract_coadd=False,
                jansky=True,
                high_pass=self.high_pass,
                match_sources=self.match_sources,
                normalize_weights=self.normalize_weights,
                compute_noisechisq=self.compute_noisechisq,
                compute_snr_annulus=self.compute_snr_annulus,
                map_id_format="{band}",
            )
            pipe.Add(_collect)
            pipe.Run()

        if not results:
            raise ValueError(
                f"spt3g.transients.TransientMapFiltering produced no output "
                f"frame for band '{band}' of {input_map.map_filename}; check "
                "that the map frame's Id matches the band string."
            )
        out_frame = results[0]
        weight_key = "Wpol" if "Wpol" in out_frame else "Wunpol"

        T_weighted = out_frame["T"]
        W = out_frame[weight_key].TT
        # Weighted-T (flux * inverse-variance) is a different physical
        # quantity than T itself, so it does not follow the plain
        # physical_value = raw / G3Units.unit convention used for T or for
        # SPTIntensityAndInverseVarianceMap's unweighted intensity; tracing
        # through ConvertMapUnitsToJansky's scale/wscale (scale ~ mJy/K per
        # pixel, wscale = 1/scale**2) shows raw weighted-T = rho_mJy_inv /
        # G3Units.mJy, i.e. rho = raw * G3Units.mJy (multiply, not divide).
        # kappa follows the same multiply convention already validated for
        # SPTIntensityAndInverseVarianceMap's inverse_variance (W * unit**2).
        # Both checked against rho/kappa => flux = T_weighted/W (unit-free
        # ratio) and snr = T_weighted/sqrt(W) (units cancel), matching the
        # established unfiltered-map convention.
        rho = enmap.enmap(np.asarray(T_weighted) * core.G3Units.mJy, wcs=T_weighted.wcs)
        kappa = enmap.enmap(np.asarray(W) * core.G3Units.mJy**2, wcs=T_weighted.wcs)

        log.info(
            "SPT3GTransientFilter.preprocess.filtered",
            shape=rho.shape,
            band=band,
        )
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
