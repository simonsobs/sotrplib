import itertools
from abc import ABC
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
from sotrplib.maps.database import FluxMapReader, IntensityMapReader, RhoKappaMapReader
from sotrplib.maps.preprocessor import MatchedFilter
from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.sources.force import (
    TwoDGaussianFitter,
)
from sotrplib.sources.sources import RegisteredSource


class ForcedPhotometryStacker(ABC):
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

    def get_coverage_maps(self, session: Session) -> None:
        map_ids = []
        for source in self.source_list:
            coord = ICRS(ra=source.position.ra, dec=source.position.dec)
            cur_maps: list[DepthOneMapTable] = get_maps_by_coverage(
                session=session, coord=coord
            )
            cur_map_ids = [m.map_id for m in cur_maps]
            map_ids.append(cur_map_ids)

        self.map_ids = map_ids

    #### Functions bellow here depreciated, to be cut

    def get_flux(
        self, sources: list[RegisteredSource], cur_map: ProcessableMap
    ) -> tuple[float, float]:  # TODO: allow different forced photometry methods
        forced_photometry = TwoDGaussianFitter(
            mode="lmfit", thumbnail_half_width=0.1 * u.deg
        )
        source_cat = RegisteredSourceCatalog(sources=sources)
        results = forced_photometry.force(input_map=cur_map, catalogs=[source_cat])

        return [result.flux.to(u.Jy).value for result in results], [
            result.err_flux.to(u.Jy).value for result in results
        ]

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


class IntensityAndInverseVarianceStacker(ForcedPhotometryStacker):
    def __init__(
        self,
        catalogs: List[RegisteredSourceCatalog],
        start_time: Time,
        end_time: Time,
        freq: str,
        array: str,
    ):
        super().__init__(
            catalogs=catalogs,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            array=array,
        )

    def get_maps(self):
        if self.map_ids is None:
            raise ValueError(
                "Map IDs have not been loaded. Call get_coverage_maps() first."
            )
        maps = []
        for cur_map_ids in self.map_ids:
            db_reader = IntensityMapReader(
                number_to_read=None,
                start_time=self.start_time,
                end_time=self.end_time,
                freq=self.freq,
                array=self.array,
                map_ids=cur_map_ids,
            )  # TODO: This is currently somewhat roundabout, as we use get_maps_by_coverage to
            # get the depth one maps that include a given point source and then filter by time, when
            # we should do both of those actions in one query.

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


class RhoKappaStacker(ForcedPhotometryStacker):
    def __init__(
        self,
        catalogs: List[RegisteredSourceCatalog],
        start_time: Time,
        end_time: Time,
        freq: str,
        array: str,
    ):
        super().__init__(
            catalogs=catalogs,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            array=array,
        )

    def get_maps(self):
        if self.map_ids is None:
            raise ValueError(
                "Map IDs have not been loaded. Call get_coverage_maps() first."
            )
        maps = []
        for cur_map_ids in self.map_ids:
            db_reader = RhoKappaMapReader(
                number_to_read=None,
                start_time=self.start_time,
                end_time=self.end_time,
                freq=self.freq,
                array=self.array,
                map_ids=cur_map_ids,
            )

            cur_maps = db_reader.map_list()
            maps.append([db_reader._build_map(m) for m in cur_maps])
        self.maps = maps

    def get_stamps(self, thumb_width=20 * u.rad):
        stamps = []
        ivar_stamps = []
        for source, cur_map in zip(self.source_list, self.maps):
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


class FluxAndSNRStacker(ForcedPhotometryStacker):
    def __init__(
        self,
        catalogs: List[RegisteredSourceCatalog],
        start_time: Time,
        end_time: Time,
        freq: str,
        array: str,
    ):
        super().__init__(
            catalogs=catalogs,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            array=array,
        )

    def get_maps(self):
        if self.map_ids is None:
            raise ValueError(
                "Map IDs have not been loaded. Call get_coverage_maps() first."
            )
        maps = []
        for cur_map_ids in self.map_ids:
            db_reader = FluxMapReader(
                number_to_read=None,
                start_time=self.start_time,
                end_time=self.end_time,
                freq=self.freq,
                array=self.array,
                map_ids=cur_map_ids,
            )

            cur_maps = db_reader.map_list()
            maps.append([db_reader._build_map(m) for m in cur_maps])
        self.maps = maps

    def get_stamps(self, thumb_width=20 * u.rad):
        stamps = []
        ivar_stamps = []
        for source, cur_map in zip(self.source_list, self.maps):
            fstamp = reproject.thumbnail(
                cur_map.flux,
                [
                    source.position.dec.to(u.rad).value,
                    source.position.ra.to(u.rad).value,
                ],
                r=thumb_width.to(u.rad).value,
                res=cur_map.flux.map_resolution.to(u.rad).value,
            )
            snr_stamp = reproject.thumbnail(
                cur_map.snr,
                [
                    source.position.dec.to(u.rad).value,
                    source.position.ra.to(u.rad).value,
                ],
                r=thumb_width.to(u.rad).value,
                res=cur_map.snr.map_resolution.to(u.rad).value,
            )
            stamps.append(fstamp)
            ivar_stamps.append(fstamp / snr_stamp)
        self.stamps = stamps
        self.ivar_stamps = ivar_stamps
