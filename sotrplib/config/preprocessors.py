from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.maps.preprocessor import (
    EdgeMask,
    GalaxyMask,
    KappaRhoCleaner,
    MapPreprocessor,
    MatchedFilter,
    PlanetMasker,
)


class PreprocessorConfig(BaseModel, ABC):
    """Abstract base for preprocessor configuration objects.

    Each subclass defines a ``preprocessor_type`` discriminator and implements
    ``to_preprocessor`` to construct a ``MapPreprocessor``.
    """

    preprocessor_type: str

    @abstractmethod
    def to_preprocessor(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPreprocessor:
        """Construct the configured preprocessor.

        Parameters
        ----------
        log : FilteringBoundLogger, optional
            Structured logger.

        Returns
        -------
        MapPreprocessor
            The constructed preprocessor instance.
        """
        return


class PlanetMaskConfig(PreprocessorConfig):
    """Configuration for the planet-masking preprocessor.

    Fields
    ------
    mask_radius : Quantity
        Radius of the circular mask around each planet (default 15 arcmin).
    """

    preprocessor_type: Literal["planet_mask"] = "planet_mask"
    mask_radius: AstroPydanticQuantity = 15 * u.arcmin

    def to_preprocessor(self, log: FilteringBoundLogger | None = None) -> PlanetMasker:
        return PlanetMasker(mask_radius=self.mask_radius, log=log)


class KappaRhoCleanerConfig(PreprocessorConfig):
    """Configuration for the kappa/rho map cleaner.

    Fields
    ------
    cut_on : {"median", "percentile", "max"}
        Reference value for the kappa threshold (default ``"median"``).
    fraction : float
        Pixels below ``fraction * reference`` are zeroed (default 0.05).
    """

    preprocessor_type: Literal["kappa_rho"] = "kappa_rho"
    cut_on: Literal["median", "percentile", "max"] = "median"
    fraction: float = 0.05

    def to_preprocessor(
        self, log: FilteringBoundLogger | None = None
    ) -> KappaRhoCleaner:
        return KappaRhoCleaner(cut_on=self.cut_on, fraction=self.fraction)


class MatchedFilterConfig(PreprocessorConfig):
    """Configuration for the matched-filter preprocessor.

    All fields mirror the parameters of ``filters.matched_filter_depth1_map``.
    See that function for full parameter descriptions.

    Fields
    ------
    infofile : Path or None
        Path to the observation info HDF file.
    maskfile : Path or None
        External mask file path.
    beam1d : Path or None
        1-D beam profile file path.
    shrink_holes : Quantity
        Hole-fill radius (default 20 arcmin).
    apod_edge : Quantity
        Edge apodization width (default 10 arcmin).
    apod_holes : Quantity
        Hole apodization width (default 5 arcmin).
    noisemask_lim : float or None
        Brightness limit for the noise mask in mJy/sr.
    noisemask_radius : Quantity
        Noise-mask radius (default 10 arcmin).
    highpass : bool
        Apply high-pass filter (default ``False``).
    band_height : Quantity
        Processing band height (default 1 degree).
    shift : float
        Shift matrix strength (default 0 = disabled).
    simple : bool
        Use power-law noise model (default ``False``).
    simple_lknee : float
        Knee multipole for the simple model (default 1000).
    simple_alpha : float
        Power-law index for the simple model (default -3.5).
    lres : tuple of int
        Noise-spectrum smoothing block size (default (70, 100)).
    pixwin : str
        Pixel window function type (default ``"nn"``).
    """

    preprocessor_type: Literal["matched_filter"] = "matched_filter"
    infofile: Path | None = None
    maskfile: Path | None = None
    beam1d: Path | None = None
    shrink_holes: AstroPydanticQuantity = 20 * u.arcmin
    apod_edge: AstroPydanticQuantity = 10 * u.arcmin
    apod_holes: AstroPydanticQuantity = 5 * u.arcmin
    noisemask_lim: float | None = None
    noisemask_radius: AstroPydanticQuantity = 10 * u.arcmin
    highpass: bool = False
    band_height: AstroPydanticQuantity = 1 * u.degree
    shift: float = 0
    simple: bool = False
    simple_lknee: float = 1000
    simple_alpha: float = -3.5
    lres: tuple = (70, 100)
    pixwin: str = "nn"

    def to_preprocessor(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPreprocessor:
        return MatchedFilter(
            infofile=self.infofile,
            maskfile=self.maskfile,
            beam1d=self.beam1d,
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
            log=log,
        )


class EdgeMaskConfig(PreprocessorConfig):
    """Configuration for the edge-masking preprocessor.

    Fields
    ------
    edge_width : Quantity[arcmin]
        Width of the masked border (default 10 arcmin).
    mask_on : str
        Map attribute used to determine the map edge (default ``"kappa"``).
    """

    preprocessor_type: Literal["edge_mask"] = "edge_mask"
    edge_width: AstroPydanticQuantity[u.arcmin] = AstroPydanticQuantity(10 * u.arcmin)
    mask_on: Literal["rho", "kappa", "flux", "snr", "inverse_variance", "intensity"] = (
        "kappa"
    )

    def to_preprocessor(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPreprocessor:
        return EdgeMask(edge_width=self.edge_width, mask_on=self.mask_on, log=log)


class GalaxyMaskConfig(PreprocessorConfig):
    """Configuration for the galactic-dust mask preprocessor.

    Fields
    ------
    mask_path : Path
        Path to the FITS galactic mask file.
    invert : bool
        If ``True``, invert the mask (default ``False``).
    """

    preprocessor_type: Literal["galaxy_mask"] = "galaxy_mask"
    mask_path: Path
    invert: bool = False

    def to_preprocessor(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPreprocessor:
        return GalaxyMask(mask_path=self.mask_path, invert=self.invert, log=log)


AllPreprocessorConfigTypes = (
    KappaRhoCleanerConfig
    | MatchedFilterConfig
    | EdgeMaskConfig
    | PlanetMaskConfig
    | GalaxyMaskConfig
)
