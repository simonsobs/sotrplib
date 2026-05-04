import itertools
from typing import List

import numpy as np
from astropy import units as u
from astropy.coordinates import ICRS
from mapcat.core import get_maps_by_coverage
from mapcat.database import DepthOneMapTable
from sqlalchemy.orm import Session

from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.sources.force import (
    TwoDGaussianFitter,
)
from sotrplib.sources.sources import RegisteredSource


class ForcedPhotometryStacker:
    def __init__(self, catalogs: List[RegisteredSourceCatalog]):
        self.catalogs = catalogs
        source_list = list(itertools.chain(*[c.sources for c in self.catalogs]))
        self.source_list = source_list

    def get_maps(self, session: Session) -> None:
        maps = []
        for source in self.source_list:
            coord = ICRS(ra=source.position.ra, dec=source.position.dec)
            cur_maps: list[DepthOneMapTable] = get_maps_by_coverage(
                session=session, coord=coord
            )
            maps.append(cur_maps)
        self.maps = maps

    def get_flux(
        self, sources: list[RegisteredSource], cur_map: DepthOneMapTable
    ) -> tuple[float, float]:  # TODO: allow different forced photometry methods
        forced_photometry = TwoDGaussianFitter(
            mode="scipy", thumbnail_half_width=0.1 * u.deg
        )
        source_cat = RegisteredSourceCatalog(sources=sources)
        results = forced_photometry.force(input_map=cur_map, catalogs=[source_cat])

        return results.flux.to(u.Jy).value, results.err_flux.to(u.Jy).value

    def stack_array(
        self, fluxes: np.ndarray, errs: np.ndarray, method: str = "inv_var"
    ) -> tuple[np.ndarray, np.ndarray]:
        if method == "inv_var":
            # Assuming err is actually err, err^2 is the variance
            ys = np.sum(fluxes / errs**2, axis=-1)  # compute numerator of yhat
            denom = np.sum(1 / errs**2, axis=-1)

            return ys / denom, 1 / denom

        else:
            raise ValueError("Error: method {} not supported.".format(method))

    def stack(self, session: Session, method: str = "inv_var"):
        fluxes = np.zeros((len(self.sources), len(self.maps)))
        errs = np.zeros((len(self.sources), len(self.maps)))
        if self.maps is None:
            self.get_maps(session=session)

        for i, cur_map in enumerate(self.maps):
            flux, err = self.get_flux(sources=self.source_list, cur_map=cur_map)
            fluxes[..., i] = flux
            errs[..., i] = err

        return self.stack_array(fluxes=fluxes, errs=errs, method=method)

        # TODO: get flux and uncertainty from results and stack
        # TODO: get stamps
