from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sources.subtractor import (
    EmptySourceSubtractor,
    PhotutilsSourceSubtractor,
    SourceSubtractor,
)


class SourceSubtractorConfig(BaseModel, ABC):
    """Abstract base for source-subtractor configuration objects.

    Each subclass defines a ``subtractor_type`` discriminator and implements
    ``to_source_subtractor`` to construct a ``SourceSubtractor``.
    """

    subtractor_type: str

    @abstractmethod
    def to_source_subtractor(
        self, log: FilteringBoundLogger | None = None
    ) -> SourceSubtractor:
        """Construct the configured source subtractor.

        Parameters
        ----------
        log : FilteringBoundLogger, optional
            Structured logger.

        Returns
        -------
        SourceSubtractor
            The constructed subtractor instance.
        """
        return


class EmptySourceSubtractorConfig(SourceSubtractorConfig):
    """Configuration for a no-op source subtractor."""

    subtractor_type: Literal["empty"] = "empty"

    def to_source_subtractor(
        self, log: FilteringBoundLogger | None = None
    ) -> EmptySourceSubtractor:
        return EmptySourceSubtractor()


class PhotutilsSourceSubtractorConfig(SourceSubtractorConfig):
    """Configuration for the photutils-based source subtractor."""

    subtractor_type: Literal["photutils"] = "photutils"

    def to_source_subtractor(
        self, log: FilteringBoundLogger | None = None
    ) -> SourceSubtractor:
        return PhotutilsSourceSubtractor()


AllSourceSubtractorConfigTypes = (
    EmptySourceSubtractorConfig | PhotutilsSourceSubtractorConfig
)
