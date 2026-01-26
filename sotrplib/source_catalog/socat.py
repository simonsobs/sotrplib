"""
Use a 'mock' database using SOCat
"""

from typing import Literal

import numpy as np
import structlog
from astropy import units as u
from astropy.coordinates import SkyCoord
from pixell import utils as pixell_utils
from pixell.wcsutils import fix_wcs
from socat.client.settings import SOCatClientSettings
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import CrossMatch, RegisteredSource
from sotrplib.utils.utils import angular_separation

from .core import SourceCatalog


class SOCatWrapper:
    def __init__(self, log: FilteringBoundLogger | None = None):
        self.log = log or structlog.get_logger()
        self.settings = SOCatClientSettings()
        self.catalog = self.settings.client
        self.log.info("socat.initialized")

    def add_source(self, source: RegisteredSource):
        self.catalog.create(
            position=SkyCoord(source.ra, source.dec),
            flux=source.flux if source.flux is not None else None,
            name=source.source_id,
        )

    def _socat_source_to_registered(self, socat_source) -> RegisteredSource:
        return RegisteredSource(
            ra=socat_source.position.ra,
            dec=socat_source.position.dec,
            flux=socat_source.flux
            if hasattr(socat_source, "flux") and (socat_source.flux is not None)
            else None,
            source_id=socat_source.name,
            crossmatches=[
                CrossMatch(
                    source_id=socat_source.name,
                    probability=1.0,
                    angular_separation=0.0 * u.deg,
                    # TBD read this from SOCat
                    frequency=90.0 * u.GHz,
                    catalog_name="mock",
                    catalog_idx=socat_source.id,
                )
            ],
        )

    def source_from_id(self, source_id) -> RegisteredSource:
        return self._socat_source_to_registered(self.catalog.get_source(id=source_id))

    def filter_source_list_to_within_map(
        self, mask_map: ProcessableMap, sources: list[RegisteredSource]
    ) -> list[RegisteredSource]:
        """
        Given a list of sources, filter to only those within the input map.
        """
        if len(sources) == 0:
            return []
        coords = SkyCoord(ra=[s.ra for s in sources], dec=[s.dec for s in sources])
        ## map may need to have wcs rotated into valid sky region.
        mask_map.flux.wcs = fix_wcs(mask_map.flux.wcs)
        y, x = mask_map.flux.wcs.world_to_pixel(coords)
        nx, ny = mask_map.flux.shape
        x, y = np.round(x).astype(int), np.round(y).astype(int)
        inside = (x >= 0) & (y >= 0) & (x < nx) & (y < ny)

        result = np.zeros_like(x, dtype=bool)
        result[inside] = np.nan_to_num(mask_map.flux[x[inside], y[inside]]).astype(bool)
        return [sources[i] for i, valid in enumerate(result) if valid]

    def get_forced_photometry_sources(
        self, minimum_flux: u.Quantity
    ) -> list[RegisteredSource]:
        return [
            self._socat_source_to_registered(x)
            for x in self.catalog.get_forced_photometry_sources(
                minimum_flux=minimum_flux
            )
        ]

    ## TODO: need to update for boxes which need to be broken in two.
    def get_sources_in_box(self, box: list[SkyCoord] | None = None):
        self.log = self.log.bind(initial_box=box)
        if box is None:
            box = [
                SkyCoord(ra=0.0 * u.deg, dec=-90.0 * u.deg),
                SkyCoord(ra=359.999 * u.deg, dec=90.0 * u.deg),
            ]
        if box[0].ra > box[1].ra and box[0].dec < box[1].dec:
            ## 0 is lower left corner and 1 is upper right corner
            ## ra increases to the left.
            box = [box[1], box[0]]

        sources_in_map = self.catalog.get_box(
            lower_left=box[0],
            upper_right=box[1],
        )
        self.log = self.log.bind(box=box, n_sources=len(sources_in_map))

        return [
            self._socat_source_to_registered(socat_source=x) for x in sources_in_map
        ]

    def get_sources_in_map(self, input_map: ProcessableMap) -> list[RegisteredSource]:
        box = input_map.bbox
        all_sources = self.get_sources_in_box(box=box)
        if len(all_sources) == 0:
            return []
        coords = SkyCoord(
            ra=[s.ra for s in all_sources], dec=[s.dec for s in all_sources]
        )
        y, x = input_map.flux.wcs.world_to_pixel(coords)
        nx, ny = input_map.flux.shape
        x, y = np.round(x).astype(int), np.round(y).astype(int)
        inside = (x >= 0) & (y >= 0) & (x < nx) & (y < ny)
        result = np.zeros_like(x, dtype=bool)
        result[inside] = np.nan_to_num(input_map.flux[x[inside], y[inside]]).astype(
            bool
        )
        return [all_sources[i] for i, valid in enumerate(result) if valid]

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
            x for x in rough_cut if angular_separation(x.ra, ra, x.dec, dec) <= radius
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


class SOCat(SourceCatalog):
    """
    A catalog implementation that uses the configured SOCat database.
    """

    def __init__(
        self,
        flux_lower_limit: u.Quantity = 0.03 * u.Jy,
        log: FilteringBoundLogger | None = None,
    ):
        self.log = log or structlog.get_logger()
        self.core = SOCatWrapper(log=self.log)
        self.flux_lower_limit = flux_lower_limit

    def add_sources(self, sources: list[RegisteredSource]):
        for source in sources:
            self.core.add_source(source=source)

    def get_sources_in_box(
        self, box: list[SkyCoord] | None = None
    ) -> list[RegisteredSource]:
        return self.core.get_sources_in_box(box=box)

    def get_sources_in_map(self, input_map: ProcessableMap) -> list[RegisteredSource]:
        return self.core.get_sources_in_map(input_map=input_map)

    def get_all_sources(self) -> list[RegisteredSource]:
        return self.core.get_sources_in_box(box=None)

    def forced_photometry_sources(self, mask_map: ProcessableMap):
        base_source_list = self.core.get_forced_photometry_sources(
            minimum_flux=self.flux_lower_limit
        )
        return self.core.filter_source_list_to_within_map(
            mask_map=mask_map,
            sources=base_source_list,
        )

    def source_by_id(self, id) -> RegisteredSource:
        return self.core.source_from_id(source_id=id)

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
            [(x.ra.to("radian").value, x.dec.to("radian").value) for x in close_sources]
        )
        if len(ra_dec_array) == 0:
            return []
        matches = pixell_utils.crossmatch(
            pos1=[[ra.to("radian").value, dec.to("radian").value]],
            pos2=ra_dec_array,
            rmax=radius.to("radian").value,
            mode=method,
            coords="radec",
        )
        sources = [close_sources[y] for _, y in matches]
        return [
            CrossMatch(
                source_id=s.source_id,
                probability=1.0 / len(sources),  ##TODO fix probability calculation
                angular_separation=angular_separation(s.ra, s.dec, ra, dec),
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
