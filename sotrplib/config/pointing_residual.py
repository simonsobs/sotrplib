from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.maps.pointing import (
    ConstantPointingOffset,
    EmptyPointingOffset,
    MapPointingOffset,
    PolynomialPointingOffset,
)


class PointingResidualConfig(BaseModel, ABC):
    """Abstract base for pointing-residual configuration objects.

    Each subclass defines a ``pointing_residual_type`` discriminator and
    implements ``to_residuals`` to construct a ``MapPointingOffset``.
    """

    pointing_residual_type: str

    @abstractmethod
    def to_residuals(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPointingOffset:
        """Construct the configured pointing-residual model.

        Parameters
        ----------
        log : FilteringBoundLogger, optional
            Structured logger.

        Returns
        -------
        MapPointingOffset
            The constructed pointing offset model.
        """
        return


class EmptyPointingResidualConfig(PointingResidualConfig):
    """Configuration for a no-op pointing model (no correction applied)."""

    pointing_residual_type: Literal["empty"] = "empty"

    def to_residuals(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPointingOffset:
        return EmptyPointingOffset()


class ConstantResidualConfig(PointingResidualConfig):
    """Configuration for a constant (per-map mean/median) pointing offset model.

    Fields
    ------
    pointing_residual_type : {"mean", "median"}
        Averaging method used to compute the offset (default ``"mean"``).
    min_snr : float
        Minimum source SNR to include in the pointing fit (default 5.0).
    min_sources : int
        Minimum number of sources required to fit a model (default 5).
    sigma_clip_level : float
        Sigma-clipping threshold for outlier rejection (default 3.0).
    """

    pointing_residual_type: Literal["mean", "median"] = "mean"
    min_snr: float = 5.0
    min_sources: int = 5
    sigma_clip_level: float = 3.0

    def to_residuals(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPointingOffset:
        return ConstantPointingOffset(
            min_snr=self.min_snr,
            min_num=self.min_sources,
            avg_method=self.pointing_residual_type,
            sigma_clip_level=self.sigma_clip_level,
            log=log,
        )


class PolynomialResidualConfig(PointingResidualConfig):
    """Configuration for a polynomial pointing offset model.

    Fields
    ------
    min_snr : float
        Minimum source SNR to include in the pointing fit (default 5.0).
    min_sources : int
        Minimum number of sources required to fit a model (default 5).
    sigma_clip_level : float
        Sigma-clipping threshold for outlier rejection (default 3.0).
    polynomial_order : int
        Order of the fitted polynomial (default 3).
    """

    pointing_residual_type: Literal["polynomial"] = "polynomial"
    min_snr: float = 5.0
    min_sources: int = 5
    sigma_clip_level: float = 3.0
    polynomial_order: int = 3

    def to_residuals(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPointingOffset:
        return PolynomialPointingOffset(
            min_snr=self.min_snr,
            min_num=self.min_sources,
            sigma_clip_level=self.sigma_clip_level,
            poly_order=self.polynomial_order,
            log=log,
        )


AllPointingResidualConfigTypes = (
    EmptyPointingResidualConfig | ConstantResidualConfig | PolynomialResidualConfig
)
