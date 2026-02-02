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
    PowerLawTransientSourceGenerator,
    SimulatedSourceGenerator,
    SOCatSourceGenerator,
)


class SourceSimulationConfig(BaseModel, ABC):
    simulation_type: str

    @abstractmethod
    def to_simulator(
        self, log: FilteringBoundLogger | None = None
    ) -> SimulatedSourceGenerator:
        return


class FixedSourceGeneratorConfig(SourceSimulationConfig):
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
    simulation_type: Literal["gaussian_transient"] = "gaussian_transient"

    flare_width_shortest: timedelta
    flare_width_longest: timedelta
    alpha_rise_min: float
    alpha_rise_max: float
    alpha_decay_min: float
    alpha_decay_max: float
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


class PowerLawSourceGeneratorConfig(SourceSimulationConfig):
    simulation_type: Literal["power_law"] = "power_law"
    flare_earliest_time: AwareDatetime
    flare_latest_time: AwareDatetime
    peak_amplitude_minimum: AstroPydanticQuantity[u.Jy]
    peak_amplitude_maximum: AstroPydanticQuantity[u.Jy]
    alpha_rise_min: float = 1.0
    alpha_rise_max: float = 3.0
    alpha_decay_min: float = -3.0
    alpha_decay_max: float = -1.0
    number: int
    smoothness_min: float = 1.0
    smoothness_max: float = 1.0
    t_ref_min: timedelta = timedelta(days=0.999)
    t_ref_max: timedelta = timedelta(days=1.001)
    catalog_fraction: float = 1.0

    def to_simulator(
        self, log: FilteringBoundLogger | None = None
    ) -> PowerLawTransientSourceGenerator:
        return PowerLawTransientSourceGenerator(
            flare_earliest_time=self.flare_earliest_time,
            flare_latest_time=self.flare_latest_time,
            peak_amplitude_minimum=self.peak_amplitude_minimum,
            peak_amplitude_maximum=self.peak_amplitude_maximum,
            alpha_rise_min=self.alpha_rise_min,
            alpha_rise_max=self.alpha_rise_max,
            alpha_decay_min=self.alpha_decay_min,
            alpha_decay_max=self.alpha_decay_max,
            number=self.number,
            smoothness_min=self.smoothness_min,
            smoothness_max=self.smoothness_max,
            t_ref_min=self.t_ref_min,
            t_ref_max=self.t_ref_max,
            catalog_fraction=self.catalog_fraction,
            log=log,
        )


AllSourceSimulationConfigTypes = (
    FixedSourceGeneratorConfig
    | GaussianTransientSourceGeneratorConfig
    | SOCatSourceGeneratorConfig
    | PowerLawSourceGeneratorConfig
)
