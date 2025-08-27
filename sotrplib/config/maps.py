"""
Map configuration
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel, Field, model_validator
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import (
    ProcessableMap,
    SimulatedMap,
    SimulatedMapFromGeometry,
    # RhoAndKappaMap,
    # IntensityAndInverseVarianceMap,
    # CoaddedMap,
    SimulationParameters,
)


class MapConfig(BaseModel, ABC):
    map_type: str

    @abstractmethod
    def to_map(self, log: FilteringBoundLogger | None = None) -> ProcessableMap:
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


AllMapConfigTypes = SimulatedMapConfig | SimulatedMapFromGeometryConfig
