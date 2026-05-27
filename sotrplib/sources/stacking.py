from abc import ABC

from astropy import units as u
from pixell import enmap, reproject

from sotrplib.maps.core import (
    ProcessableMap,
)
from sotrplib.maps.database import (
    MapCatDatabaseReader,
)
from sotrplib.maps.preprocessor import MapPreprocessor
from sotrplib.sources.sources import RegisteredSource


class Stamps:
    """
    Class to hold stamps, ivar stamps, and times for a single source.

    Attributes:
    -----------
    flux_stamps: list[enmap.ndmap]
        List of flux stamps for the source.
    ivar_stamps: list[enmap.ndmap]
        List of ivar stamps for the source.
    times: list[float]
        List of times corresponding to each stamp.
    """

    def __init__(
        self,
        flux_stamps: list[list[enmap.ndmap]],
        ivar_stamps: list[list[enmap.ndmap]],
        times: list[float],
    ):
        self.flux_stamps = flux_stamps
        self.ivar_stamps = ivar_stamps
        self.times = times


class Stacker(ABC):
    """
    Stacking class to handle the the creation of flux and ivar stamps for a list of sources.

    Attributes:
    -----------
    reader: MapCatDatabaseReader
        Reader to read maps from the database.
    preprocessors: list[MapPreprocessor]
        List of preprocessors to apply to the maps after reading and before stamping.
    maps: list[ProcessableMap]
        List of maps read from the database and preprocessed, ready for stamping. Starts as none and set by the _read_maps method.

    Methods:
    --------
    _read_maps():
        Reads maps from the database using the reader, applies preprocessors, and finalizes the maps for stamping.
    stamp(sources: list[RegisteredSource], thumb_width=20*u.arcmin) -> Stamps:
        Creates flux and ivar stamps for a list of sources.
    """

    def __init__(
        self,
        reader: MapCatDatabaseReader,  # TODO: Support readers other than MapCatDatabaseReader
        preprocessors: list[MapPreprocessor] | None = None,
    ):
        self.reader = reader
        self.preprocessors = preprocessors or []
        self.maps = None

    def _read_maps(self) -> list[ProcessableMap]:
        maps = self.reader.map_list()
        for imap in maps:
            imap.build()

        if self.preprocessors:
            # Note preprocessors can change the type of the map, so we need to check the type after preprocessing
            for imap in maps:
                for preprocessor in self.preprocessors:
                    imap = preprocessor.preprocess(imap)

        for imap in maps:
            imap.finalize()

        self.maps = maps

    def stamp(
        self, sources: list[RegisteredSource], thumb_width=20 * u.arcmin
    ) -> Stamps:
        stamps = []
        for source in sources:
            cur_stamps = []
            cur_ivar_stamps = []
            cur_times = []
            if self.maps is None:
                self._read_maps()
            for cur_map in self.maps:
                fstamp = reproject.thumbnails(
                    cur_map.flux,
                    [
                        source.dec.to(u.rad).value,
                        source.ra.to(u.rad).value,
                    ],
                    r=thumb_width.to(u.rad).value,
                )
                snrstamp = reproject.thumbnails(
                    cur_map.snr,
                    [
                        source.dec.to(u.rad).value,
                        source.ra.to(u.rad).value,
                    ],
                    r=thumb_width.to(u.rad).value,
                )
                tstamp = reproject.thumbnails(
                    cur_map.time_mean,
                    [
                        source.dec.to(u.rad).value,
                        source.ra.to(u.rad).value,
                    ],
                    r=thumb_width.to(u.rad).value,
                )
                time = tstamp.at([0, 0])
                cur_stamps.append(fstamp)
                cur_ivar_stamps.append(fstamp / snrstamp)
                cur_times.append(time)

            stamps.append(
                Stamps(
                    flux_stamps=cur_stamps, ivar_stamps=cur_ivar_stamps, times=cur_times
                )
            )

        return stamps
