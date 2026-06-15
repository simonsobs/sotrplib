"""
Map Coadding configuration
"""

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.maps.map_coadding import EmptyMapCoadder, MapCoadder, RhoKappaMapCoadder


class MapCoadderConfig(BaseModel, ABC):
    """Abstract base for map-coadder configuration objects.

    Each subclass defines a ``coadd_type`` discriminator and implements
    ``to_coadder`` to construct a ``MapCoadder``.
    """

    coadd_type: str

    @abstractmethod
    def to_coadder(
        self,
        log: FilteringBoundLogger | None = None,
    ) -> MapCoadder:
        """Construct the configured map coadder.

        Parameters
        ----------
        log : FilteringBoundLogger, optional
            Structured logger.

        Returns
        -------
        MapCoadder
            The constructed coadder instance.
        """
        return


class EmptyMapCoadderConfig(MapCoadderConfig):
    """Configuration for a no-op coadder that returns each map unchanged."""

    coadd_type: Literal["empty"] = "empty"

    def to_coadder(self, log: FilteringBoundLogger | None = None) -> MapCoadder:
        return EmptyMapCoadder()


class RhoKappaMapCoadderConfig(MapCoadderConfig):
    """Configuration for the rho/kappa map coadder.

    Fields
    ------
    frequencies : list of str or None
        Frequency bands to coadd (e.g. ``["f090", "f150"]``).  If ``None``,
        all unique frequencies in the input are used.
    arrays : list of str or None
        Arrays to coadd over.  Defaults to ``["coadd"]`` (all arrays merged).
    instrument : str or None
        Instrument name assigned to the output coadd map.
    """

    coadd_type: Literal["rhokappa"] = "rhokappa"

    frequencies: list[str] | None = None
    arrays: list[str] | None = None
    instrument: str | None = None

    def to_coadder(
        self,
        log: FilteringBoundLogger | None = None,
    ) -> MapCoadder:
        return RhoKappaMapCoadder(
            frequencies=self.frequencies,
            arrays=self.arrays,
            instrument=self.instrument,
            log=log,
        )


AllMapCoadderConfigTypes = RhoKappaMapCoadderConfig | EmptyMapCoadderConfig
