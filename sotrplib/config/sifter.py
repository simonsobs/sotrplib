from abc import ABC, abstractmethod
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sifter.core import (
    DefaultSifter,
    EmptySifter,
    SiftingProvider,
    SimpleCatalogSifter,
)
from sotrplib.sifter.transient_crossmatch import TransientCrossmatcher
from sotrplib.source_catalog.external_catalogs import (
    ExternalCatalog,
    GaiaCatalog,
    TableExternalCatalog,
)


class ExternalCatalogConfig(BaseModel, ABC):
    catalog_type: str

    @abstractmethod
    def to_catalog(self, log: FilteringBoundLogger | None = None) -> ExternalCatalog:
        return


class MillionQuasarCatalogConfig(ExternalCatalogConfig):
    catalog_type: Literal["million_quasar"] = "million_quasar"
    path: str
    density_radius: AstroPydanticQuantity[u.deg] = 1.0 * u.deg

    def to_catalog(
        self, log: FilteringBoundLogger | None = None
    ) -> TableExternalCatalog:
        return TableExternalCatalog.from_millionquasar_file(
            path=self.path,
            density_radius=self.density_radius,
            log=log,
        )


class GaiaCatalogConfig(ExternalCatalogConfig):
    catalog_type: Literal["gaia"] = "gaia"
    density_radius: AstroPydanticQuantity[u.arcmin] = 5.0 * u.arcmin
    max_mag: float | None = None
    row_limit: int = 2000

    def to_catalog(self, log: FilteringBoundLogger | None = None) -> GaiaCatalog:
        return GaiaCatalog(
            density_radius=self.density_radius,
            max_mag=self.max_mag,
            row_limit=self.row_limit,
            log=log,
        )


class TableCatalogConfig(ExternalCatalogConfig):
    """
    Generic file-backed catalog, e.g. a nearby galaxy catalog
    (GLADE+, HECATE, 2MRS) with a user-supplied column mapping.
    """

    catalog_type: Literal["table"] = "table"
    path: str
    ra_column: str
    dec_column: str
    name_column: str
    coord_unit: tuple[str, str] = ("deg", "deg")
    mag_column: str | None = None
    redshift_column: str | None = None
    distance_column: str | None = None
    distance_unit: str = "Mpc"
    source_type: str = "galaxy"
    catalog_name: str | None = None
    density_radius: AstroPydanticQuantity[u.deg] = 1.0 * u.deg

    def to_catalog(
        self, log: FilteringBoundLogger | None = None
    ) -> TableExternalCatalog:
        return TableExternalCatalog.from_table_file(
            path=self.path,
            ra_column=self.ra_column,
            dec_column=self.dec_column,
            name_column=self.name_column,
            coord_unit=self.coord_unit,
            mag_column=self.mag_column,
            redshift_column=self.redshift_column,
            distance_column=self.distance_column,
            distance_unit=self.distance_unit,
            source_type=self.source_type,
            catalog_name=self.catalog_name,
            density_radius=self.density_radius,
            log=log,
        )


AllExternalCatalogConfigTypes = (
    MillionQuasarCatalogConfig | GaiaCatalogConfig | TableCatalogConfig
)


class TransientCrossmatchConfig(BaseModel):
    catalogs: list[AllExternalCatalogConfigTypes] = []
    match_radius: AstroPydanticQuantity[u.arcmin] = 2.0 * u.arcmin
    position_error_scale: float = 3.0
    type_priors: dict[str, float] | None = None
    min_score: float = 0.0
    set_source_type: bool = True

    def to_crossmatcher(
        self, log: FilteringBoundLogger | None = None
    ) -> TransientCrossmatcher:
        return TransientCrossmatcher(
            catalogs=[c.to_catalog(log=log) for c in self.catalogs],
            match_radius=self.match_radius,
            position_error_scale=self.position_error_scale,
            type_priors=self.type_priors,
            min_score=self.min_score,
            set_source_type=self.set_source_type,
            log=log,
        )


class SifterConfig(BaseModel, ABC):
    sifter_type: str

    @abstractmethod
    def to_sifter(self, log: FilteringBoundLogger | None = None) -> SiftingProvider:
        return


class EmptySifterConfig(SifterConfig):
    sifter_type: Literal["empty"] = "empty"

    def to_sifter(self, log: FilteringBoundLogger | None = None) -> EmptySifter:
        return EmptySifter()


class SimpleCatalogSifterConfig(SifterConfig):
    sifter_type: Literal["simple"] = "simple"
    radius: AstroPydanticQuantity[u.arcmin] = 1.0 * u.arcmin
    method: Literal["closest", "all"] = "closest"
    transient_crossmatch: TransientCrossmatchConfig | None = None

    def to_sifter(self, log: FilteringBoundLogger | None = None) -> SimpleCatalogSifter:
        return SimpleCatalogSifter(
            radius=self.radius,
            method=self.method,
            transient_crossmatcher=self.transient_crossmatch.to_crossmatcher(log=log)
            if self.transient_crossmatch is not None
            else None,
            log=log,
        )


class DefaultSifterConfig(SifterConfig):
    sifter_type: Literal["default"] = "default"
    min_match_radius: AstroPydanticQuantity[u.arcmin] = 1.5 * u.arcmin
    transient_crossmatch: TransientCrossmatchConfig | None = None

    def to_sifter(self, log: FilteringBoundLogger | None = None) -> SiftingProvider:
        return DefaultSifter(
            min_match_radius=self.min_match_radius,
            transient_crossmatcher=self.transient_crossmatch.to_crossmatcher(log=log)
            if self.transient_crossmatch is not None
            else None,
            log=log,
        )


AllSifterConfigTypes = (
    EmptySifterConfig | DefaultSifterConfig | SimpleCatalogSifterConfig
)
