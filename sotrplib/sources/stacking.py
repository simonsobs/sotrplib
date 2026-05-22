from abc import ABC

import numpy as np
from astropy import units as u
from pixell import reproject

from sotrplib.maps.core import (
    ProcessableMap,
    RhoAndKappaMap,
)
from sotrplib.maps.database import (
    FluxMapReader,
    MapCatDatabaseReader,
)
from sotrplib.maps.preprocessor import MapPreprocessor
from sotrplib.sources.sources import RegisteredSource


class Stacker(ABC):
    def __init__(
        self,
        reader: MapCatDatabaseReader,  # TODO: Support readers other than MapCatDatabaseReader
        preprocessors: list[MapPreprocessor] | None = None,
    ):
        self.reader = reader
        self.preprocessors = preprocessors or []

    def _read_maps(self) -> list[RhoAndKappaMap]:
        maps = self.reader.map_list()
        for imap in maps:
            imap.build()

        if type(self.reader) is FluxMapReader:
            for imap in maps:
                imap.ivar = imap.flux / imap.snr
                imap.kappa = imap.ivar
                imap.rho = imap.flux * imap.kappa
                imap = RhoAndKappaMap()

        if self.preprocessors:
            for imap in maps:
                for preprocessor in self.preprocessors:
                    imap = preprocessor.preprocess(imap)

        if not np.all([type(imap) is RhoAndKappaMap for imap in maps]):
            raise ValueError(
                "Not all maps are of type RhoAndKappaMap. Check your applied Preprocessors."
            )

        return maps

    def stamp(
        self, sources: list[RegisteredSource], thumb_width=20 * u.arcmin
    ) -> list[list[ProcessableMap]]:
        stamps = []
        ivar_stamps = []
        for source in sources:
            cur_stamps = []
            cur_ivar_stamps = []
            for cur_map in self._read_maps():
                rstamp = reproject.thumbnails(
                    cur_map.rho,
                    [
                        source.dec.to(u.rad).value,
                        source.ra.to(u.rad).value,
                    ],
                    r=thumb_width.to(u.rad).value,
                )
                kstamp = reproject.thumbnails(
                    cur_map.kappa,
                    [
                        source.dec.to(u.rad).value,
                        source.ra.to(u.rad).value,
                    ],
                    r=thumb_width.to(u.rad).value,
                )
                cur_stamps.append(rstamp / kstamp)
                cur_ivar_stamps.append(kstamp)
            stamps.append(cur_stamps)
            ivar_stamps.append(cur_ivar_stamps)

        return stamps, ivar_stamps


"""
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
            coord = ICRS(ra=source.position().ra, dec=source.position().dec)
            cur_maps: list[DepthOneMapTable] = get_maps_by_coverage(
                session=session, coord=coord
            )
            cur_map_ids = [m.map_id for m in cur_maps]
            map_ids.append(cur_map_ids)

        self.map_ids = map_ids

class IntensityIvarStacker(ForcedPhotometryStacker):
    def __init__(
        self,
        sources: list[RegisteredSource],
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
            rerun=True,
        )

        self.maps = db_reader.map_list()
        for imap in self.maps:
            imap.build()



    def get_stamps(self, thumb_width=20 * u.arcmin):
        if self.filtered_maps is None:
            raise ValueError("Maps have not been filtered. Call filter_maps() first.")
        stamps = []
        ivar_stamps = []
        for source in self.sources:
            cur_stamps = []
            cur_ivar_stamps = []
            for cur_map in self.filtered_maps:
                rstamp = reproject.thumbnails(
                    cur_map.rho,
                    [
                        source.dec.to(u.rad).value,
                        source.ra.to(u.rad).value,
                    ],
                    r=thumb_width.to(u.rad).value,
                )
                kstamp = reproject.thumbnails(
                    cur_map.kappa,
                    [
                        source.dec.to(u.rad).value,
                        source.ra.to(u.rad).value,
                    ],
                    r=thumb_width.to(u.rad).value,
                )
                cur_stamps.append(rstamp / kstamp)
                cur_ivar_stamps.append(kstamp)
            stamps.append(cur_stamps)
            ivar_stamps.append(cur_ivar_stamps)
        self.stamps = stamps
        self.ivar_stamps = ivar_stamps


class RhoKappaStacker(ForcedPhotometryStacker):
    def __init__(
        self,
        sources: list[RegisteredSource],
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
            rerun=True,
        )

        self.maps = db_reader.map_list()
        for imap in self.maps:
            imap.build()

    def get_stamps(self, thumb_width=20 * u.arcmin):
        stamps = []
        ivar_stamps = []
        for source in self.sources:
            cur_stamps = []
            cur_ivar_stamps = []
            for cur_map in self.maps:
                rstamp = reproject.thumbnails(
                    cur_map.rho,
                    [
                        source.dec.to(u.rad).value,
                        source.ra.to(u.rad).value,
                    ],
                    r=thumb_width.to(u.rad).value,
                )
                kstamp = reproject.thumbnails(
                    cur_map.kappa,
                    [
                        source.dec.to(u.rad).value,
                        source.ra.to(u.rad).value,
                    ],
                    r=thumb_width.to(u.rad).value,
                )
                cur_stamps.append(rstamp / kstamp)
                cur_ivar_stamps.append(kstamp)
            stamps.append(cur_stamps)
            ivar_stamps.append(cur_ivar_stamps)
        self.stamps = stamps
        self.ivar_stamps = ivar_stamps


class FluxSNRStacker(ForcedPhotometryStacker):
    def __init__(
        self,
        sources: list[RegisteredSource],
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
            rerun=True,
        )

        self.maps = db_reader.map_list()
        for imap in self.maps:
            imap.build()

    def get_stamps(self, thumb_width=20 * u.arcmin):
        stamps = []
        ivar_stamps = []
        for source in self.sources:
            cur_stamps = []
            cur_ivar_stamps = []
            for cur_map in self.maps:
                fstamp = reproject.thumbnails(
                    cur_map.flux,
                    [
                        source.dec.to(u.rad).value,
                        source.ra.to(u.rad).value,
                    ],
                    r=thumb_width.to(u.rad).value,
                )
                snr_stamp = reproject.thumbnails(
                    cur_map.snr,
                    [
                        source.dec.to(u.rad).value,
                        source.ra.to(u.rad).value,
                    ],
                    r=thumb_width.to(u.rad).value,
                )
                cur_stamps.append(fstamp)
                cur_ivar_stamps.append(fstamp / snr_stamp)
            stamps.append(cur_stamps)
            ivar_stamps.append(cur_ivar_stamps)
        self.stamps = stamps
        self.ivar_stamps = ivar_stamps

"""
