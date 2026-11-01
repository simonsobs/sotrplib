from abc import ABC, abstractmethod
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.maps.pointing import (
    ConstantPointingOffset,
    EmptyPointingOffset,
    MapPointingOffset,
    PolynomialPointingOffset,
)
from sotrplib.maps.robust_pointing import RobustPointingOffset


class PointingResidualConfig(BaseModel, ABC):
    pointing_residual_type: str

    @abstractmethod
    def to_residuals(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPointingOffset:
        return


class EmptyPointingResidualConfig(PointingResidualConfig):
    pointing_residual_type: Literal["empty"] = "empty"

    def to_residuals(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPointingOffset:
        return EmptyPointingOffset()


class ConstantResidualConfig(PointingResidualConfig):
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


class RobustResidualConfig(PointingResidualConfig):
    """
    Robust pointing residual. See ``sotrplib.maps.robust_pointing``.

    "robust_constant" fits one offset for the map. "robust_gp" fits one
    offset plus a smooth correction that goes back to the constant offset in
    regions without sources.
    """

    pointing_residual_type: Literal["robust_constant", "robust_gp"] = "robust_gp"
    min_snr: float = 5.0
    min_sources: int = 3
    max_offset: AstroPydanticQuantity[u.arcmin] = 5.0 * u.arcmin
    max_model_offset: AstroPydanticQuantity[u.arcmin] = 5.0 * u.arcmin
    clip_sigma: float = 4.0
    sys_floor: AstroPydanticQuantity[u.arcsec] = 2.0 * u.arcsec
    fwhm: AstroPydanticQuantity[u.arcmin] | None = None
    length_scale: AstroPydanticQuantity[u.deg] = 5.0 * u.deg
    amplitude: AstroPydanticQuantity[u.arcsec] = 2.5 * u.arcsec
    max_correction: AstroPydanticQuantity[u.arcsec] = 15.0 * u.arcsec

    def to_residuals(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPointingOffset:
        return RobustPointingOffset(
            method="gp" if self.pointing_residual_type == "robust_gp" else "constant",
            min_snr=self.min_snr,
            min_num=self.min_sources,
            max_offset=self.max_offset,
            max_model_offset=self.max_model_offset,
            clip_sigma=self.clip_sigma,
            sys_floor=self.sys_floor,
            fwhm=self.fwhm,
            length_scale=self.length_scale,
            amplitude=self.amplitude,
            max_correction=self.max_correction,
            log=log,
        )


AllPointingResidualConfigTypes = (
    EmptyPointingResidualConfig
    | ConstantResidualConfig
    | PolynomialResidualConfig
    | RobustResidualConfig
)
