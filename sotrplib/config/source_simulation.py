from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sims.sources.core import (
    RandomSourceSimulation,
    RandomSourceSimulationParameters,
    SourceSimulation,
)


class SourceSimulationConfig(BaseModel, ABC):
    simulation_type: str

    @abstractmethod
    def to_simulator(self, log: FilteringBoundLogger | None = None) -> SourceSimulation:
        return


class RandomSourceSimulationConfig(SourceSimulationConfig):
    simulation_type: Literal["random"] = "random"
    parameters: RandomSourceSimulationParameters

    def to_simulator(
        self, log: FilteringBoundLogger | None = None
    ) -> RandomSourceSimulation:
        return RandomSourceSimulation(parameters=self.parameters, log=log)


AllSourceSimulationConfigTypes = RandomSourceSimulationConfig
