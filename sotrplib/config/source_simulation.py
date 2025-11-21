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


AllSourceSimulationConfigTypes = (
    FixedSourceGeneratorConfig | GaussianTransientSourceGeneratorConfig
)
