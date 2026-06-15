from abc import ABC, abstractmethod
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sources.force import (
    EmptyForcedPhotometry,
    ForcedPhotometryProvider,
    SimpleForcedPhotometry,
    TwoDGaussianFitter,
    TwoDGaussianPointingFitter,
)


class ForcedPhotometryConfig(BaseModel, ABC):
    """Abstract base for forced-photometry configuration objects.

    Each subclass defines a ``photometry_type`` discriminator and implements
    ``to_forced_photometry`` to construct a ``ForcedPhotometryProvider``.
    """

    photometry_type: str

    @abstractmethod
    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> ForcedPhotometryProvider:
        """Construct the configured forced-photometry provider.

        Parameters
        ----------
        log : FilteringBoundLogger, optional
            Structured logger.

        Returns
        -------
        ForcedPhotometryProvider
            The constructed provider instance.
        """
        return


class EmptyPhotometryConfig(ForcedPhotometryConfig):
    """Configuration for a no-op forced-photometry provider."""

    photometry_type: Literal["empty"] = "empty"

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> EmptyForcedPhotometry:
        return EmptyForcedPhotometry()


class SimpleForcedPhotometryConfig(ForcedPhotometryConfig):
    """Configuration for simple map-lookup forced photometry.

    Fields
    ------
    mode : {"spline", "nn"}
        Interpolation mode used when sampling the flux map.
    """

    photometry_type: Literal["simple"] = "simple"
    mode: Literal["spline", "nn"]

    def to_forced_photometry(self, log=None) -> SimpleForcedPhotometry:
        return SimpleForcedPhotometry(mode=self.mode, log=log)


class LmfitGaussianFitterConfig(ForcedPhotometryConfig):
    """Configuration for the lmfit 2-D Gaussian forced-photometry fitter.

    Fields
    ------
    flux_limit_centroid : Quantity[Jy]
        Minimum flux to attempt centroid fitting (default 0.3 Jy).
    reproject_thumbnails : bool
        Reproject thumbnails before fitting (default ``False``).
    thumbnail_half_width : Quantity[deg]
        Half-width of the fitting thumbnail (default 0.1 deg).
    allowable_center_offset : Quantity[arcmin]
        Maximum allowed centroid offset from the prior position (default 1 arcmin).
    near_source_rel_flux_limit : float
        Relative flux limit for nearby sources included in the fit (default 1.0).
    goodness_of_fit_threshold : float or None
        Chi-squared threshold above which fits are rejected.
    """

    photometry_type: Literal["lmfit"] = "lmfit"
    flux_limit_centroid: AstroPydanticQuantity[u.Jy] = u.Quantity(0.3, "Jy")
    reproject_thumbnails: bool = False
    thumbnail_half_width: AstroPydanticQuantity[u.deg] = u.Quantity(0.1, "deg")
    allowable_center_offset: AstroPydanticQuantity[u.arcmin] = u.Quantity(1.0, "arcmin")
    near_source_rel_flux_limit: float = 1.0
    goodness_of_fit_threshold: float | None = None

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> TwoDGaussianFitter:
        return TwoDGaussianFitter(
            mode="lmfit",
            flux_limit_centroid=self.flux_limit_centroid,
            reproject_thumbnails=self.reproject_thumbnails,
            thumbnail_half_width=self.thumbnail_half_width,
            allowable_center_offset=self.allowable_center_offset,
            near_source_rel_flux_limit=self.near_source_rel_flux_limit,
            goodness_of_fit_threshold=self.goodness_of_fit_threshold,
            log=log,
        )


class Lmfit2DGaussianPointingConfig(ForcedPhotometryConfig):
    """Configuration for the lmfit 2-D Gaussian pointing fitter.

    Like ``LmfitGaussianFitterConfig`` but intended for pointing sources —
    uses a looser centroid offset and a stricter relative flux limit.

    Fields
    ------
    min_flux : Quantity[Jy]
        Minimum source flux to attempt fitting (default 0.3 Jy).
    reproject_thumbnails : bool
        Reproject thumbnails before fitting (default ``False``).
    thumbnail_half_width : Quantity[deg]
        Half-width of the fitting thumbnail (default 0.1 deg).
    allowable_center_offset : Quantity[arcmin]
        Maximum allowed centroid offset (default 3 arcmin).
    near_source_rel_flux_limit : float
        Relative flux limit for nearby sources (default 0.3).
    goodness_of_fit_threshold : float or None
        Chi-squared threshold above which fits are rejected.
    """

    photometry_type: Literal["lmfit_pointing"] = "lmfit_pointing"
    min_flux: AstroPydanticQuantity[u.Jy] = u.Quantity(0.3, "Jy")
    reproject_thumbnails: bool = False
    thumbnail_half_width: AstroPydanticQuantity[u.deg] = u.Quantity(0.1, "deg")
    allowable_center_offset: AstroPydanticQuantity[u.arcmin] = u.Quantity(3.0, "arcmin")
    near_source_rel_flux_limit: float = 0.3
    goodness_of_fit_threshold: float | None = None

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> TwoDGaussianPointingFitter:
        return TwoDGaussianPointingFitter(
            mode="lmfit",
            min_flux=self.min_flux,
            reproject_thumbnails=self.reproject_thumbnails,
            thumbnail_half_width=self.thumbnail_half_width,
            allowable_center_offset=self.allowable_center_offset,
            near_source_rel_flux_limit=self.near_source_rel_flux_limit,
            goodness_of_fit_threshold=self.goodness_of_fit_threshold,
            log=log,
        )


AllForcedPhotometryConfigTypes = (
    EmptyPhotometryConfig
    | SimpleForcedPhotometryConfig
    | LmfitGaussianFitterConfig
    | Lmfit2DGaussianPointingConfig
)
