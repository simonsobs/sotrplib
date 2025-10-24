"""
Use a 'mock' database using SOCat
"""

from pathlib import Path
from typing import Literal

import numpy as np
import structlog
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from pixell import utils as pixell_utils
from socat.client.core import ClientBase
from socat.client.mock import Client as SOCatMockClient
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import CrossMatch, RegisteredSource
from sotrplib.utils.utils import angular_separation

from .core import SourceCatalog


class SOCatWrapper:
    """
    Helper that wraps SOCat and converts its internal functions
    into those reading and producing RegisteredSource objects.
    """

    def __init__(
        self, log: FilteringBoundLogger | None = None, catalog: ClientBase | None = None
    ):
        self.catalog = catalog or SOCatMockClient()
        self.log = log or structlog.get_logger()
        self.log.info("socat.initialized")

    def add_source(self, source: RegisteredSource):
        self.catalog.create(
            ra=source.ra.to_value(u.deg),
            dec=source.dec.to_value(u.deg),
            name=source.source_id,
        )

    def _socat_source_to_registered(self, socat_source) -> RegisteredSource:
        return RegisteredSource(
            ra=socat_source.ra * u.deg,
            dec=socat_source.dec * u.deg,
            source_id=socat_source.name,
            crossmatches=[
                CrossMatch(
                    source_id=socat_source.name,
                    probability=1.0,
                    distance=0.0 * u.deg,
                    # TBD read this from SOCat
                    frequency=90.0 * u.GHz,
                    catalog_name="mock",
                    catalog_idx=socat_source.id,
                )
            ],
        )

    def source_from_id(self, source_id) -> RegisteredSource:
        return self._socat_source_to_registered(self.catalog.get_source(id=source_id))

    ## TODO: this doesn't work... maybe need source_catalog/core.py to handle boxes that cross RA=0
    def get_sources_in_box(self, box: list[SkyCoord] | None = None):
        if box is None:
            box = [
                SkyCoord(ra=-180 * u.deg, dec=-90 * u.deg),
                SkyCoord(ra=180 * u.deg, dec=90 * u.deg),
            ]
        if box[0].ra > box[1].ra and box[0].dec < box[1].dec:
            ## 0 is lower left corner and 1 is upper right corner
            ## ra increases to the left.
            box = [box[1], box[0]]
        if box[1].ra > 180 * u.deg and box[0].ra < 180 * u.deg:
            ## socat expects -180,180 limits.
            ## if we cross the 180 line, need to split into two boxes
            self.log.info("socat.get_sources_in_box.splitting_box", box=box)
            box1 = [
                SkyCoord(ra=box[0].ra, dec=box[0].dec),
                SkyCoord(ra=180 * u.deg, dec=box[1].dec),
            ]
            box2 = [
                SkyCoord(ra=-180 * u.deg, dec=box[0].dec),
                SkyCoord(ra=box[1].ra - 360 * u.deg, dec=box[1].dec),
            ]
            sources1 = self.get_sources_in_box(box=box1)
            sources2 = self.get_sources_in_box(box=box2)
            self.log.info(
                "socat.get_sources_in_box.splitting_box",
                box=box,
                box1=box1,
                box2=box2,
                n_sources1=len(sources1),
                n_sources2=len(sources2),
            )
            sources_in_map = sources1 + sources2
        else:
            ra_min = box[0].ra.to(u.deg).value
            ra_min = ra_min if ra_min <= 180 else ra_min - 360
            ra_max = box[1].ra.to(u.deg).value
            ra_max = ra_max if ra_max <= 180 else ra_max - 360
            dec_min = np.min([box[0].dec.to(u.deg).value, box[1].dec.to(u.deg).value])
            dec_max = np.max([box[0].dec.to(u.deg).value, box[1].dec.to(u.deg).value])
            box = [[ra_min, dec_min], [ra_max, dec_max]]
            ## astropy SkyCoord uses 0 to 360 convention for RA, but SOCat uses -180 to 180
            sources_in_map = self.catalog.get_box(
                ra_min=ra_min,
                ra_max=ra_max,
                dec_min=dec_min,
                dec_max=dec_max,
            )
        self.log.info(
            "socat.get_sources_in_box", box=box, n_sources=len(sources_in_map)
        )

        return [
            self._socat_source_to_registered(socat_source=x) for x in sources_in_map
        ]

    def nearby_sources(
        self, ra: u.Quantity, dec: u.Quantity, radius: u.Quantity = 0.1 * u.deg
    ) -> list[RegisteredSource]:
        rough_cut = self.get_sources_in_box(
            box=[
                SkyCoord(ra=ra - radius, dec=dec - radius),
                SkyCoord(ra=ra + radius, dec=dec + radius),
            ]
        )

        filtered = [
            self._socat_source_to_registered(socat_source=x)
            for x in rough_cut
            if angular_separation(x.ra, ra, x.dec, dec) <= radius
        ]

        self.log.info(
            "socat.nearby_sources",
            rough=len(rough_cut),
            final=len(filtered),
            ra=ra,
            dec=dec,
            radius=radius,
        )

        return filtered


class SOCatFITSCatalog(SourceCatalog):
    """
    A catalog, using SOCat under the hood, that reads a FITS file from disk.
    """

    def __init__(
        self,
        path: Path | None = None,
        hdu: int = 1,
        flux_lower_limit: u.Quantity = 0.01 * u.Jy,
        log: FilteringBoundLogger | None = None,
    ):
        self.log = log or structlog.get_logger()
        self.path = path
        self.hdu = hdu
        self.flux_lower_limit = flux_lower_limit
        self.core = SOCatWrapper(log=self.log)
        self.valid_fluxes = set()
        self._read_fits_file()

    def _read_fits_file(self):
        if self.path is None:
            self.log.info("socat_fits.intialized_empty")
            return

        data = fits.open(self.path)[self.hdu]

        # TODO: Remove this when it's part of SOCat
        for _, row in enumerate(data.data):
            flux = row["fluxJy"] * u.Jy
            if flux < self.flux_lower_limit:
                continue
            ra = (
                row["raDeg"] if row["raDeg"] < 180.0 else row["raDeg"] - 360.0
            ) * u.deg
            dec = row["decDeg"] * u.deg
            name = row["name"]

            # TODO: Need to store flux in SOCat, otherwise we can't filter on it.
            self.core.catalog.create(
                ra=ra.to_value(u.deg),
                dec=dec.to_value(u.deg),
                name=name,
            )

            if flux > self.flux_lower_limit:
                self.valid_fluxes.add(name)

        self.log.info("socat_fits.loaded", n_sources=len(self.valid_fluxes))

    def add_sources(self, sources: list[RegisteredSource]):
        for source in sources:
            self.core.add_source(source=source)
            if source.flux >= self.flux_lower_limit:
                self.valid_fluxes.add(source.source_id)

    def get_sources_in_box(
        self, box: list[SkyCoord] | None = None
    ) -> list[RegisteredSource]:
        return self.core.get_sources_in_box(box=box)

    def get_sources_in_map(self, input_map: ProcessableMap) -> list[RegisteredSource]:
        map_bounds = input_map.bbox()
        self.log.info(
            "socat_fits.get_sources_in_map",
            map_bounds_deg=map_bounds.to(u.deg),
        )
        # if ra_min is greater than ra_max, we are crossing the 0 point
        # so we need to split the box into two boxes.
        # since pixell uses -180 to 180 convention, we need to go 0 to ra_min then ra_max to 0
        if map_bounds[0].ra > map_bounds[1].ra:
            box1 = [
                SkyCoord(map_bounds[0].ra, 0 * u.rad),
                SkyCoord(map_bounds[1].ra, map_bounds[0].dec),
            ]
            box2 = [
                SkyCoord(map_bounds[0].ra, map_bounds[1].dec),
                SkyCoord(map_bounds[1].ra, 0 * u.rad),
            ]
            sources1 = self.core.get_sources_in_box(box=box1)
            sources2 = self.core.get_sources_in_box(box=box2)
            return sources1 + sources2
        else:
            return self.core.get_sources_in_box(box=map_bounds)

    def get_all_sources(self) -> list[RegisteredSource]:
        sources = self.core.get_sources_in_box(
            box=[
                SkyCoord(-179.99 * u.deg, 89.99 * u.deg),
                SkyCoord(179.99 * u.deg, -89.99 * u.deg),
            ]
        )
        return sources

    def forced_photometry_sources(self, box: list[SkyCoord] | None = None):
        return [
            x for x in self.get_sources_in_box(box) if x.source_id in self.valid_fluxes
        ]

    def source_by_id(self, id) -> RegisteredSource:
        return self.core.source_from_id(id=id)

    def crossmatch(
        self,
        ra: u.Quantity,
        dec: u.Quantity,
        radius: u.Quantity,
        method: Literal["closest", "all"],
    ) -> list[CrossMatch]:
        """
        Get sources within radius of the catalog.
        """
        ra_min = ra - 2.0 * radius
        ra_max = ra + 2.0 * radius
        dec_min = dec - 2.0 * radius
        dec_max = dec + 2.0 * radius
        close_sources = self.get_sources_in_box(
            [
                SkyCoord(ra=ra_min, dec=dec_min),
                SkyCoord(ra=ra_max, dec=dec_max),
            ],
        )
        ra_dec_array = np.asarray(
            [(x.ra.to("deg").value, x.dec.to("deg").value) for x in close_sources]
        )
        if len(ra_dec_array) == 0:
            return []
        print(ra_dec_array)
        matches = pixell_utils.crossmatch(
            pos1=[[ra.to("deg").value, dec.to("deg").value]],
            pos2=ra_dec_array,
            rmax=radius.to("deg").value,
            mode=method,
            coords="radec",
        )
        print(matches)
        sources = [close_sources[y] for _, y in matches]
        return [
            CrossMatch(
                source_id=s.source_id,
                probability=1.0 / len(sources),  ##TODO fix probability calculation
                distance=angular_separation(s.ra, ra, s.dec, dec),
                flux=s.flux,
                err_flux=s.err_flux,
                frequency=s.frequency,
                catalog_name=s.catalog_name,
                catalog_idx=y,
                alternate_names=s.alternate_names,
                ra=s.ra,
                dec=s.dec,
            )
            for y, s in enumerate(sources)
        ]
