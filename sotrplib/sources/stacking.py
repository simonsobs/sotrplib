from abc import ABC

import numpy as np
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
    """Flux stamps, inverse-variance stamps, and times for a single source.

    Attributes
    ----------
    flux_stamps : list of enmap.ndmap
        Flux cutouts, one per map visit.
    ivar_stamps : list of enmap.ndmap
        Inverse-variance cutouts, one per map visit.
    times : list of float
        Observation times corresponding to each stamp.
    """

    def __init__(
        self,
        flux_stamps: list[enmap.ndmap],
        ivar_stamps: list[enmap.ndmap],
        times: list[float],
    ):
        self.flux_stamps = flux_stamps
        self.ivar_stamps = ivar_stamps
        self.times = times


class Stacker(ABC):
    """Abstract stacker that cuts flux and ivar stamps for a list of sources.

    Parameters
    ----------
    reader : MapCatDatabaseReader
        Reader used to load maps from the database.
    preprocessors : list of MapPreprocessor, optional
        Pipeline steps applied to each map after reading and before stamping.

    Attributes
    ----------
    maps : list of ProcessableMap
        Map list obtained from the reader; populated on construction.
    """

    def __init__(
        self,
        reader: MapCatDatabaseReader,  # TODO: Support readers other than MapCatDatabaseReader
        preprocessors: list[MapPreprocessor] | None = None,
    ):
        self.reader = reader
        self.preprocessors = preprocessors or []
        self.maps = self.reader.map_list()

    def _read_map(self, imap: ProcessableMap) -> ProcessableMap:
        imap.build()

        if self.preprocessors:
            # Note preprocessors can change the type of the map, so we need to check the type after preprocessing
            for preprocessor in self.preprocessors:
                imap = preprocessor.preprocess(imap)

        imap.finalize()

        return imap

    def stamp(
        self, sources: list[RegisteredSource], thumb_width=20 * u.arcmin
    ) -> Stamps:
        stamps = np.empty((len(sources), 3, len(self.maps)), dtype=object)
        for j, cur_map in enumerate(self.maps):
            cur_map = self._read_map(cur_map)
            for i, source in enumerate(sources):
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
                stamps[i, 0, j] = fstamp
                stamps[i, 1, j] = (snrstamp / fstamp) ** 2
                stamps[i, 2, j] = time

        stamps_list = []

        for i in range(len(stamps)):
            cur_stamps = stamps[i, 0, :].tolist()
            cur_ivar_stamps = stamps[i, 1, :].tolist()
            cur_times = stamps[i, 2, :].tolist()
            stamp = Stamps(
                flux_stamps=cur_stamps, ivar_stamps=cur_ivar_stamps, times=cur_times
            )
            stamps_list.append(stamp)

        return stamps_list
