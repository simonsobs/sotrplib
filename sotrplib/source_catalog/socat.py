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
                    frequency=90.0 * u.Ghz,
                    catalog_name="mock",
                    catalog_idx=socat_source.id,
                )
            ],
        )

    def source_from_id(self, source_id) -> RegisteredSource:
        return self._socat_source_to_registered(self.catalog.get_source(id=source_id))

    def sources_in_box(self, box: list[SkyCoord] | None = None):
        sources_in_map = self.catalog.get_box(
            ra_min=box[0][1].to(u.deg).value,
            ra_max=box[1][1].to(u.deg).value,
            dec_min=box[0][0].to(u.deg).value,
            dec_max=box[1][0].to(u.deg).value,
        )

        self.log.info("socat.sources_in_box", box=box, n_sources=len(sources_in_map))

        return [
            self._socat_source_to_registered(socat_source=x) for x in sources_in_map
        ]

    def nearby_sources(
        self, ra: u.Quantity, dec: u.Quantity, radius: u.Quantity = 0.1 * u.deg
    ) -> list[RegisteredSource]:
        rough_cut = self.sources_in_box(
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
        path: Path,
        hdu: int = 1,
        flux_lower_limit: u.Quantity = 0.01 * u.Jy,
        log: FilteringBoundLogger | None = None,
    ):
        self.log = log or structlog.get_logger()
        self.path = path
        self.hdu = hdu
        self.flux_lower_limit = flux_lower_limit
        self.core = SOCatWrapper(log=self.log)

        self._read_fits_file()

    def _read_fits_file(self):
        data = fits.open(self.path)[self.hdu]

        # TODO: Remove this when it's part of SOCat
        self.valid_fluxes = set()

        for index, row in enumerate(data.data):
            flux = row["fluxJy"] * u.Jy
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

        self.log.info("socat_fits.loaded", n_sources=index + 1)

    def add_sources(self, sources: list[RegisteredSource]):
        for source in sources:
            self.core.add_source(source=source)

    def sources_in_box(
        self, box: list[SkyCoord] | None = None
    ) -> list[RegisteredSource]:
        return self.core.sources_in_box(box=box)

    def forced_photometry_sources(self, box: list[SkyCoord] | None = None):
        return [x for x in self.sources_in_box(box) if x.source_id in self.valid_fluxes]

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

        close_sources = self.sources_in_box(
            self,
            [
                SkyCoord(ra=ra - 2.0 * radius, dec=dec - 2.0 * radius),
                SkyCoord(ra=ra + 2.0 * radius, dec=dec + 2.0 * radius),
            ],
        )

        ra_dec_array = np.asarray(
            [(x.ra.to("deg").value, x.dec.to("deg").value) for x in close_sources]
        )

        matches = pixell_utils.crossmatch(
            pos1=[[ra.to("deg").value, dec.to("deg").value]],
            pos2=ra_dec_array,
            rmax=radius.to("arcmin").value,
            mode=method,
        )

        sources = [close_sources[y] for _, y in matches]

        return [
            CrossMatch(
                source_id=s.source_id,
                probability=1.0 / len(sources),
                distance=angular_separation(s.ra, ra, s.dec, dec),
                flux=s.flux,
                err_flux=s.err_flux,
                frequency=s.frequency,
                catalog_name=s.catalog_name,
                catalog_idx=y,
                alternate_names=s.alternate_names,
            )
            for s, y in zip(sources, matches)
        ]
