"""
Map configuration
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity, AstroPydanticUnit
from pydantic import BaseModel, Field, model_validator
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import (
    IntensityAndInverseVarianceMap,
    ProcessableMap,
    RhoAndKappaMap,
)
from sotrplib.maps.database import MapCatDatabaseReader
from sotrplib.sims.maps import (
    SimulatedMap,
    SimulatedMapFromGeometry,
    SimulationParameters,
)


class MapConfig(BaseModel, ABC):
    map_type: str

    @abstractmethod
    def to_map(self, log: FilteringBoundLogger | None = None) -> ProcessableMap:
        return


class MapGeneratorConfig(BaseModel, ABC):
    map_generator_type: str

    @abstractmethod
    def to_generator(
        self, log: FilteringBoundLogger | None = None
    ) -> Iterable[ProcessableMap]:
        return


class SimulatedMapConfig(MapConfig):
    map_type: Literal["simulated"] = "simulated"
    observation_start: datetime
    observation_end: datetime
    frequency: str = "f090"
    array: str = "pa5"
    simulation_parameters: SimulationParameters = Field(
        default_factory=SimulationParameters
    )

    def to_map(self, log: FilteringBoundLogger | None = None) -> SimulatedMap:
        return SimulatedMap(
            observation_start=self.observation_start,
            observation_end=self.observation_end,
            frequency=self.frequency,
            array=self.array,
            simulation_parameters=self.simulation_parameters,
            log=log,
        )


class SimulatedMapFromGeometryConfig(MapConfig):
    map_type: Literal["simulated_geometry"] = "simulated_geometry"
    resolution: AstroPydanticQuantity[u.arcmin]
    geometry_source_map: Path
    observation_start: datetime | None
    observation_end: datetime | None
    time_map_filename: Path | None
    frequency: str = "f090"
    array: str = "pa5"
    map_noise: AstroPydanticQuantity[u.Jy] = u.Quantity(0.01, "Jy")

    @model_validator(mode="after")
    def check_time_or_map(self):
        if (
            self.time_map_filename is None
            and self.observation_start is None
            and self.observation_end is None
        ):
            raise ValueError(
                "One of time_map_filename or start_time and end_time must be provided"
            )

    def to_map(
        self, log: FilteringBoundLogger | None = None
    ) -> SimulatedMapFromGeometry:
        return SimulatedMapFromGeometry(
            resolution=self.resolution,
            geometry_source_map=self.geometry_source_map,
            observation_start=self.observation_start,
            observation_end=self.observation_end,
            time_map_filename=self.time_map_filename,
            frequency=self.frequency,
            array=self.array,
            map_noise=self.map_noise,
            log=log,
        )


class RhoKappaMapConfig(MapConfig):
    map_type: Literal["filtered"] = "filtered"
    rho_map_path: Path
    kappa_map_path: Path
    time_map_path: Path | None = None
    frequency: str | None = "f090"
    array: str | None = "pa5"
    observation_start: datetime | None = None
    observation_end: datetime | None = None
    box: AstroPydanticQuantity[u.deg] | None = None
    flux_units: AstroPydanticUnit = u.Unit("Jy")

    def to_map(self, log: FilteringBoundLogger | None = None) -> RhoAndKappaMap:
        return RhoAndKappaMap(
            rho_filename=self.rho_map_path,
            kappa_filename=self.kappa_map_path,
            start_time=self.observation_start,
            end_time=self.observation_end,
            time_filename=self.time_map_path,
            box=self.box,
            frequency=self.frequency,
            array=self.array,
            flux_units=self.flux_units,
            log=log,
        )


class InverseVarianceMapConfig(MapConfig):
    map_type: Literal["inverse_variance"] = "inverse_variance"
    intensity_map_path: Path
    weights_map_path: Path
    time_map_path: Path | None = None
    frequency: str | None = "f090"
    array: str | None = "pa5"
    observation_start: datetime | None = None
    observation_end: datetime | None = None
    box: AstroPydanticQuantity[u.deg] | None = None
    intensity_units: AstroPydanticUnit = u.Unit("K")

    def to_map(
        self, log: FilteringBoundLogger | None = None
    ) -> IntensityAndInverseVarianceMap:
        return IntensityAndInverseVarianceMap(
            intensity_filename=self.intensity_map_path,
            inverse_variance_filename=self.weights_map_path,
            time_filename=self.time_map_path,
            start_time=self.observation_start,
            end_time=self.observation_end,
            box=self.box,
            frequency=self.frequency,
            array=self.array,
            intensity_units=self.intensity_units,
            log=log,
        )


class MapCatDatabaseConfig(MapConfig):
    map_generator_type: Literal["mapcat_database"] = "mapcat_database"
    number_to_read: int = 1

    def to_generator(
        self, log: FilteringBoundLogger | None = None
    ) -> Iterable[ProcessableMap]:
        return MapCatDatabaseReader(
            number_to_read=self.number_to_read,
            log=log,
        )


AllMapConfigTypes = (
    SimulatedMapConfig
    | SimulatedMapFromGeometryConfig
    | RhoKappaMapConfig
    | InverseVarianceMapConfig
)

AllMapGeneratorConfigTypes = MapCatDatabaseConfig | list[AllMapConfigTypes]
