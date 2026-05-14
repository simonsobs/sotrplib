import itertools
from typing import List

import numpy as np
from astropy import units as u
from astropy.coordinates import ICRS
from astropy.time import Time
from mapcat.core import get_maps_by_coverage
from mapcat.database import DepthOneMapTable
from pixell import reproject
from sqlalchemy.orm import Session

from sotrplib.maps.core import (
    MatchedFilteredIntensityAndInverseVarianceMap,
    ProcessableMap,
)
from sotrplib.maps.database import IntensityMapReader
from sotrplib.maps.preprocessor import MatchedFilter
from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.sources.force import (
    TwoDGaussianFitter,
)
from sotrplib.sources.sources import RegisteredSource


class ForcedPhotometryStacker:
    def __init__(
        self,
        catalogs: List[RegisteredSourceCatalog],
        start_time: Time,
        end_time: Time,
        freq: str,
        array: str,
    ):
        if end_time < start_time:
            raise ValueError("End time must be after start time.")
        self.catalogs = catalogs
        source_list = list(itertools.chain(*[c.sources for c in self.catalogs]))
        self.source_list = source_list
        self.start_time = start_time
        self.end_time = end_time
        self.freq = freq
        self.array = array

    def get_maps(self, session: Session) -> None:
        maps = []
        for source in self.source_list:
            coord = ICRS(ra=source.position.ra, dec=source.position.dec)
            cur_maps: list[DepthOneMapTable] = get_maps_by_coverage(
                session=session, coord=coord
            )
            map_ids = [m.map_id for m in cur_maps]
            db_reader = IntensityMapReader(
                number_to_read=len(cur_maps),
                start_time=self.start_time,
                end_time=self.end_time,
                freq=self.freq,
                array=self.array,
                map_ids=map_ids,
            )  # TODO: This is currently somewhat roundabout, as we use get_maps_by_coverage to
            # get the depth one maps that include a given point source and then filter by time, when
            # we should do both of those actions in one query.

            # Below is probably not going to work since IntensityMapReader has it's own DB session used to do
            # map_list, but we've got our own session here. This could maybe be fixed by splitting into two functions?
            # Alternatively I'm just doing too much shit in this object and should take a more functional approach.
            cur_maps = db_reader.map_list()
            maps.append([db_reader._build_map(m) for m in cur_maps])
        self.maps = maps

    def filter_maps(self):
        if self.maps is None:
            raise ValueError("Maps have not been loaded. Call get_maps() first.")
        filtered_maps = []
        preprocessor = MatchedFilter()
        for imap in self.maps:
            if imap is None or len(imap) == 0:
                raise ValueError(
                    "No maps found for source {source_id}.".format(
                        source_id=imap.source_id
                    )
                )

            filtered_map: MatchedFilteredIntensityAndInverseVarianceMap = (
                preprocessor.preprocess(imap)
            )
            filtered_maps.append(filtered_map)

        self.filtered_maps = filtered_maps

    def get_stamps(self, thumb_width=20 * u.rad):
        if self.filtered_maps is None:
            raise ValueError("Maps have not been filtered. Call filter_maps() first.")
        stamps = []
        ivar_stamps = []
        for source, cur_map in zip(self.source_list, self.filtered_maps):
            rstamp = reproject.thumbnail(
                cur_map.rho,
                [
                    source.position.dec.to(u.rad).value,
                    source.position.ra.to(u.rad).value,
                ],
                r=thumb_width.to(u.rad).value,
                res=cur_map.rho.map_resolution.to(u.rad).value,
            )
            kstamp = reproject.thumbnail(
                cur_map.kappa,
                [
                    source.position.dec.to(u.rad).value,
                    source.position.ra.to(u.rad).value,
                ],
                r=thumb_width.to(u.rad).value,
                res=cur_map.kappa.map_resolution.to(u.rad).value,
            )
            stamps.append(rstamp / kstamp)
            ivar_stamps.append(kstamp)
        self.stamps = stamps
        self.ivar_stamps = ivar_stamps

    def get_flux(
        self, sources: list[RegisteredSource], cur_map: ProcessableMap
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
