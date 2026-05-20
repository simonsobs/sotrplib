from abc import ABC
from typing import List

from astropy import units as u
from astropy.coordinates import ICRS
from astropy.time import Time
from mapcat.core import get_maps_by_coverage
from mapcat.database import DepthOneMapTable
from pixell import reproject
from sqlalchemy.orm import Session

from sotrplib.maps.core import (
    MatchedFilteredIntensityAndInverseVarianceMap,
)
from sotrplib.maps.database import FluxMapReader, IntensityMapReader, RhoKappaMapReader
from sotrplib.maps.preprocessor import MatchedFilter
from sotrplib.sources.sources import RegisteredSource


class ForcedPhotometryStacker(ABC):
    def __init__(
        self,
        sources: list[RegisteredSource],
        start_time: Time,
        end_time: Time,
        freq: str,
        array: str,
    ):
        if end_time < start_time:
            raise ValueError("End time must be after start time.")
        self.sources = sources
        self.start_time = start_time
        self.end_time = end_time
        self.freq = freq
        self.array = array

    def get_coverage_maps(self, session: Session) -> None:
        map_ids = []
        for source in self.sources:
            coord = ICRS(ra=source.position.ra, dec=source.position.dec)
            cur_maps: list[DepthOneMapTable] = get_maps_by_coverage(
                session=session, coord=coord
            )
            cur_map_ids = [m.map_id for m in cur_maps]
            map_ids.append(cur_map_ids)

        self.map_ids = map_ids


class IntensityAndInverseVarianceStacker(ForcedPhotometryStacker):
    def __init__(
        self,
        sources: List[RegisteredSource],
        start_time: Time,
        end_time: Time,
        freq: str,
        array: str,
    ):
        super().__init__(
            sources=sources,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            array=array,
        )

    def get_maps(self):
        db_reader = IntensityMapReader(
            number_to_read=None,
            start_time=self.start_time,
            end_time=self.end_time,
            sources=self.sources,
            frequency=self.freq,
            array=self.array,
        )

        self.maps = db_reader.map_list()

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
        for source, cur_map in zip(self.sources, self.filtered_maps):
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
        sources: List[RegisteredSource],
        start_time: Time,
        end_time: Time,
        freq: str,
        array: str,
    ):
        super().__init__(
            sources=sources,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            array=array,
        )

    def get_maps(self):
        db_reader = RhoKappaMapReader(
            number_to_read=None,
            start_time=self.start_time,
            end_time=self.end_time,
            sources=self.sources,
            frequency=self.freq,
            array=self.array,
        )

        self.maps = db_reader.map_list()

    def get_stamps(self, thumb_width=20 * u.rad):
        stamps = []
        ivar_stamps = []
        for source, cur_map in zip(self.sources, self.maps):
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
        sources: List[RegisteredSource],
        start_time: Time,
        end_time: Time,
        freq: str,
        array: str,
    ):
        super().__init__(
            sources=sources,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            array=array,
        )

    def get_maps(self):
        db_reader = FluxMapReader(
            number_to_read=None,
            start_time=self.start_time,
            end_time=self.end_time,
            sources=self.sources,
            frequency=self.freq,
            array=self.array,
        )

        self.maps = db_reader.map_list()

    def get_stamps(self, thumb_width=20 * u.rad):
        stamps = []
        ivar_stamps = []
        for source, cur_map in zip(self.sources, self.maps):
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
