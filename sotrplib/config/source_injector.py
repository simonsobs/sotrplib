from abc import ABC, abstractmethod
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sims.source_injector import (
    EmptySourceInjector,
    PhotutilsSourceInjector,
    SourceInjector,
)


class SourceInjectorConfig(BaseModel, ABC):
    """Abstract base for source-injector configuration objects.

    Each subclass defines an ``injector_type`` discriminator and implements
    ``to_injector`` to construct a ``SourceInjector``.
    """

    injector_type: str

    @abstractmethod
    def to_injector(self, log: FilteringBoundLogger | None = None) -> SourceInjector:
        """Construct the configured source injector.

        Parameters
        ----------
        log : FilteringBoundLogger, optional
            Structured logger.

        Returns
        -------
        SourceInjector
            The constructed injector instance.
        """
        return


class EmptySourceInjectorConfig(SourceInjectorConfig):
    """Configuration for a no-op source injector."""

    injector_type: Literal["empty"] = "empty"

    def to_injector(
        self, log: FilteringBoundLogger | None = None
    ) -> EmptySourceInjector:
        return EmptySourceInjector()


class PhotutilsSourceInjectorConfig(SourceInjectorConfig):
    """Configuration for the photutils Gaussian source injector.

    Fields
    ------
    gauss_fwhm : Quantity[arcmin]
        Nominal beam FWHM for injected Gaussian sources (default 2.2 arcmin).
    gauss_theta_min, gauss_theta_max : Quantity[deg]
        Range of orientation angles drawn uniformly (default 0–90 deg).
    fwhm_uncertainty_fraction : float
        Fractional scatter applied to the FWHM of each injected source
        (default 0.01).
    progress_bar : bool
        Show a tqdm progress bar during injection (default ``False``).
    """

    injector_type: Literal["photutils"] = "photutils"
    gauss_fwhm: AstroPydanticQuantity[u.arcmin] = 2.2 * u.arcmin
    gauss_theta_min: AstroPydanticQuantity[u.deg] = 0 * u.deg
    gauss_theta_max: AstroPydanticQuantity[u.deg] = 90 * u.deg
    fwhm_uncertainty_fraction: float = 0.01
    progress_bar: bool = False

    def to_injector(
        self, log: FilteringBoundLogger | None = None
    ) -> PhotutilsSourceInjector:
        return PhotutilsSourceInjector(
            gauss_fwhm=self.gauss_fwhm,
            gauss_theta_min=self.gauss_theta_min,
            gauss_theta_max=self.gauss_theta_max,
            fwhm_uncertainty_fraction=self.fwhm_uncertainty_fraction,
            progress_bar=self.progress_bar,
            log=log,
        )


AllSourceInjectorConfigTypes = EmptySourceInjectorConfig | PhotutilsSourceInjectorConfig
