from pathlib import Path
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity

from sotrplib.config.source_catalog import SourceCatalogConfig
from sotrplib.source_catalog.solar_system_object_catalog import (
    SolarSystemObjectCatalog,
    SSOCat,
)


class SSOCatalogConfig(SourceCatalogConfig):
    catalog_type: Literal["sso"] = "sso"
    db_path: Path | None = None
    observer_lat: AstroPydanticQuantity = -22.96098 * u.deg
    observer_lon: AstroPydanticQuantity = -67.7876 * u.deg
    observer_elev: AstroPydanticQuantity = 5180 * u.m

    def to_source_catalog(self, log=None) -> SolarSystemObjectCatalog:
        from sotrplib.solar_system.solar_system import create_observer

        observer = create_observer(
            lat=self.observer_lat,
            lon=self.observer_lon,
            elev=self.observer_elev,
        )

        return SSOCat(
            db_path=self.db_path,
            observer=observer,
        )


AllSSOCatalogConfigTypes = SSOCatalogConfig
