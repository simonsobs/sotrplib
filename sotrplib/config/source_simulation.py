from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import AwareDatetime, BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sims.sim_source_generators import (
    FixedSourceGenerator,
    GaussianTransientSourceGenerator,
    SimulatedSourceGenerator,
    SOCatSourceGenerator,
)


class SourceSimulationConfig(BaseModel, ABC):
    """Abstract base for source-simulation configuration objects.

    Each subclass defines a ``simulation_type`` discriminator and implements
    ``to_simulator`` to construct a ``SimulatedSourceGenerator``.
    """

    simulation_type: str

    @abstractmethod
    def to_simulator(
        self, log: FilteringBoundLogger | None = None
    ) -> SimulatedSourceGenerator:
        """Construct the configured source generator.

        Parameters
        ----------
        log : FilteringBoundLogger, optional
            Structured logger.

        Returns
        -------
        SimulatedSourceGenerator
            The constructed generator instance.
        """
        return


class FixedSourceGeneratorConfig(SourceSimulationConfig):
    """Configuration for a fixed-flux simulated source generator.

    Fields
    ------
    min_flux, max_flux : Quantity[Jy]
        Flux range for uniformly-drawn simulated sources.
    number : int
        Number of sources to simulate.
    catalog_fraction : float
        Fraction of simulated sources placed at known catalog positions
        (default 1.0).
    """

    simulation_type: Literal["fixed"] = "fixed"
    min_flux: AstroPydanticQuantity[u.Jy]
    max_flux: AstroPydanticQuantity[u.Jy]
    number: int
    catalog_fraction: float = 1.0

    def to_simulator(
        self, log: FilteringBoundLogger | None = None
    ) -> FixedSourceGenerator:
        return FixedSourceGenerator(
            min_flux=self.min_flux,
            max_flux=self.max_flux,
            number=self.number,
            catalog_fraction=self.catalog_fraction,
            log=log,
        )


class GaussianTransientSourceGeneratorConfig(SourceSimulationConfig):
    """Configuration for a Gaussian-profile transient source generator.

    Fields
    ------
    flare_width_shortest, flare_width_longest : timedelta
        Range of flare durations drawn uniformly.
    peak_amplitude_minimum, peak_amplitude_maximum : Quantity[Jy]
        Range of peak flare amplitudes drawn uniformly.
    number : int
        Number of transient sources to simulate.
    flare_earliest_time, flare_latest_time : AwareDatetime or None
        Time window within which flare peaks are placed.
    catalog_fraction : float
        Fraction placed at known catalog positions (default 1.0).
    """

    simulation_type: Literal["gaussian_transient"] = "gaussian_transient"

    flare_width_shortest: timedelta
    flare_width_longest: timedelta
    peak_amplitude_minimum: AstroPydanticQuantity[u.Jy]
    peak_amplitude_maximum: AstroPydanticQuantity[u.Jy]
    number: int
    flare_earliest_time: AwareDatetime | None = None
    flare_latest_time: AwareDatetime | None = None
    catalog_fraction: float = 1.0

    def to_simulator(
        self, log: FilteringBoundLogger | None = None
    ) -> GaussianTransientSourceGenerator:
        return GaussianTransientSourceGenerator(
            flare_earliest_time=self.flare_earliest_time,
            flare_latest_time=self.flare_latest_time,
            flare_width_shortest=self.flare_width_shortest,
            flare_width_longest=self.flare_width_longest,
            peak_amplitude_minimum=self.peak_amplitude_minimum,
            peak_amplitude_maximum=self.peak_amplitude_maximum,
            number=self.number,
            catalog_fraction=self.catalog_fraction,
            log=log,
        )


class SOCatSourceGeneratorConfig(SourceSimulationConfig):
    """Configuration for a mixed fixed/transient source generator seeded from SOCat.

    Fields
    ------
    fraction_fixed : float
        Fraction of catalog sources simulated as fixed (default 0.5).
    fraction_gaussian : float
        Fraction simulated as Gaussian transients (default 0.5).
    flare_earliest_time, flare_latest_time : AwareDatetime
        Time window for transient flare peaks.
    flare_width_shortest, flare_width_longest : timedelta
        Range of flare durations.
    peak_amplitude_minimum_factor, peak_amplitude_maximum_factor : float
        Flare amplitude as a multiple of the catalog source flux.
    """

    simulation_type: Literal["socat"] = "socat"
    fraction_fixed: float = 0.5
    fraction_gaussian: float = 0.5
    flare_earliest_time: AwareDatetime
    flare_latest_time: AwareDatetime
    flare_width_shortest: timedelta
    flare_width_longest: timedelta
    peak_amplitude_minimum_factor: float = 1.0
    peak_amplitude_maximum_factor: float = 5.0

    def to_simulator(
        self, log: FilteringBoundLogger | None = None
    ) -> SOCatSourceGenerator:
        return SOCatSourceGenerator(
            fraction_fixed=self.fraction_fixed,
            fraction_gaussian=self.fraction_gaussian,
            flare_earliest_time=self.flare_earliest_time,
            flare_latest_time=self.flare_latest_time,
            flare_width_shortest=self.flare_width_shortest,
            flare_width_longest=self.flare_width_longest,
            peak_amplitude_minimum_factor=self.peak_amplitude_minimum_factor,
            peak_amplitude_maximum_factor=self.peak_amplitude_maximum_factor,
            log=log,
        )


AllSourceSimulationConfigTypes = (
    FixedSourceGeneratorConfig
    | GaussianTransientSourceGeneratorConfig
    | SOCatSourceGeneratorConfig
)
