from abc import ABC, abstractmethod

import numpy as np
import structlog
from astropy import units as u
from pixell import enmap, reproject
from structlog.types import FilteringBoundLogger

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
        flux_stamps: list[enmap.ndmap],
        ivar_stamps: list[enmap.ndmap],
        times: list[float],
    ):
        self.flux_stamps = flux_stamps
        self.ivar_stamps = ivar_stamps
        self.times = times


class StackedStamp:
    """
    Class to hold a stacked flux stamp and its corresponding ivar stamp.
    This is essentially just a single stamp but for clarity we define a separate class for it.
    Also unlike a stamp it does not have a time associated with it, since it is a combination of multiple stamps.

    Attributes:
    -----------
    flux: enmap.ndmap
        The stacked flux stamp.
    ivar: enmap.ndmap
        The corresponding ivar stamp for the stacked flux.
    """

    def __init__(self, flux: enmap.ndmap, ivar: enmap.ndmap):
        self.flux = flux
        self.ivar = ivar


def clean_stamps(stamps: Stamps, log: FilteringBoundLogger | None = None) -> Stamps:
    """
    Clean the stamps by removing any stamps with non-finite times or any
    points with zero/NaN ivar. Also removes any stamps with non-finite flux.

    Parameters:
    -----------
    stamps: Stamps
        The Stamps object to clean.
    log: FilteringBoundLogger | None
        Optional logger for logging information. If not provided, a default logger will be used.

    Returns:
    --------
    Stamps
        A new Stamps object with only the usable stamps.
    """
    log = (log or structlog.get_logger()).bind(func="clean_stamps")
    n_bad_time = 0
    n_bad_pixels = 0
    cleaned_flux_stamps = []
    cleaned_ivar_stamps = []
    cleaned_times = []

    for flux_stamp, ivar_stamp, time in zip(
        stamps.flux_stamps, stamps.ivar_stamps, stamps.times
    ):
        if not np.isfinite(time):
            n_bad_time += 1
            continue
        if not (
            np.all(np.isfinite(flux_stamp))
            and np.all(np.isfinite(ivar_stamp))
            and np.all(ivar_stamp > 0)
        ):
            n_bad_pixels += 1
            continue

        cleaned_flux_stamps.append(flux_stamp)
        cleaned_ivar_stamps.append(ivar_stamp)
        cleaned_times.append(time)

    n_input = len(stamps.flux_stamps)
    n_kept = len(cleaned_flux_stamps)

    log.debug(
        "clean_stamps.summary",
        n_input=n_input,
        n_kept=n_kept,
        n_bad_time=n_bad_time,
        n_bad_pixels=n_bad_pixels,
    )

    if n_kept == 0 and n_input > 0:
        log.warning("clean_stamps.no_usable_stamps", n_input=n_input)
    return Stamps(
        flux_stamps=cleaned_flux_stamps,
        ivar_stamps=cleaned_ivar_stamps,
        times=cleaned_times,
    )


class StackingMethod(ABC):
    @abstractmethod
    def combine(
        self, flux: np.ndarray, ivar: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """flux, ivar shaped (N, ny, nx) -> stacked (ny, nx) flux and ivar."""

    def stack(
        self, stamps: Stamps, log: FilteringBoundLogger | None = None
    ) -> list[StackedStamp]:
        """
        Stack the stamps. Combine controls the actual stacking method,
        and the stack function handles the cleaning of the stamps and the creation of the StackedStamp object.
        In general this function should not be overridden, but rather the combine function should be overridden to implement a new stacking method.

        Parameters:
        -----------
        stamps: Stamps
            The Stamps object containing the flux and ivar stamps to be stacked.
        log: FilteringBoundLogger | None
            Optional logger for logging information. If not provided, a default logger will be used.

        Returns:
        --------
        list[StackedStamp]
            A list containing a single StackedStamp object with the stacked flux and ivar stamps.
        """
        log = (log or structlog.get_logger()).bind(method=type(self).__name__)
        stamps = clean_stamps(stamps=stamps, log=log)
        if not stamps.flux_stamps:
            return []
        wcs = stamps.flux_stamps[0].wcs
        flux = np.stack(stamps.flux_stamps)
        ivar = np.stack(stamps.ivar_stamps)
        f, w = self.combine(flux, ivar)
        log.debug("StackingMethod.stack.stacked", n_stamps=len(stamps.flux_stamps))
        return [StackedStamp(enmap.enmap(f, wcs), enmap.enmap(w, wcs))]


class MeanStacking(StackingMethod):
    def combine(
        self, flux: np.ndarray, ivar: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Compute the simple mean stack.
        # Return (mean_flux, mean_ivar).
        mean_flux = np.mean(flux, axis=0)
        mean_ivar = len(ivar) ** 2 / np.sum(1 / ivar, axis=0)
        return mean_flux, mean_ivar


class InverseVarianceStacking(StackingMethod):
    def combine(
        self, flux: np.ndarray, ivar: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Compute the inverse variance weighted mean stack.
        # Return (weighted_mean_flux, weighted_mean_ivar).
        weighted_mean_flux = np.sum(flux * ivar, axis=0) / np.sum(ivar, axis=0)
        weighted_mean_ivar = np.sum(ivar, axis=0)
        return weighted_mean_flux, weighted_mean_ivar


class Stacker:
    """
    Stacking class to handle the the creation of flux and ivar stamps for a list of sources.

    Attributes:
    -----------
    reader: MapCatDatabaseReader
        Reader to read maps from the database.
    preprocessors: list[MapPreprocessor]
        List of preprocessors to apply to the maps after reading and before stamping.
    maps: list[ProcessableMap]
        List of maps read from the database and preprocessed, ready for stamping.

    Methods:
    --------
    _read_map():
        Reads maps from the database using the reader, applies preprocessors, and finalizes the maps for stamping.
    stamp(sources: list[RegisteredSource], thumb_width=20*u.arcmin) -> list[Stamps]:
        Creates flux and ivar stamps for a list of sources.
    stack(sources: list[RegisteredSource], method: StackingMethod, thumb_width=20*u.arcmin) -> list[list[StackedStamp]]:
        Stacks the flux and ivar stamps for a list of sources using the specified stacking method
    """

    def __init__(
        self,
        reader: MapCatDatabaseReader,  # TODO: Support readers other than MapCatDatabaseReader
        preprocessors: list[MapPreprocessor] | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.reader = reader
        self.preprocessors = preprocessors or []
        self.maps = self.reader.map_list()

        self.stamps: list[Stamps] | None = None  # Will be set by the stamp() method
        self._stamped_sources: list[RegisteredSource] | None = (
            None  # Will be set by the stamp() method
        )
        self._stamped_thumb_width: u.Quantity | None = (
            None  # Will be set by the stamp() method
        )

        self.log = log or structlog.get_logger()

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
    ) -> list[Stamps]:
        log = self.log.bind(func="Stacker.stamp")
        log.info(
            "Stacker.stamp.start",
            n_sources=len(sources),
            n_maps=len(self.maps),
            thumb_width=thumb_width,
        )
        stamps = np.empty((len(sources), 3, len(self.maps)), dtype=object)
        for j, raw_map in enumerate(self.maps):
            cur_map = self._read_map(raw_map)
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

            log.debug("Stacker.stamp.map_done", map_index=j)
            cur_map.release()
            raw_map.release()

        stamps_list = []

        for i in range(len(stamps)):
            cur_stamps = stamps[i, 0, :].tolist()
            cur_ivar_stamps = stamps[i, 1, :].tolist()
            cur_times = stamps[i, 2, :].tolist()
            stamp = Stamps(
                flux_stamps=cur_stamps, ivar_stamps=cur_ivar_stamps, times=cur_times
            )
            stamps_list.append(stamp)

        self.stamps = stamps_list
        self._stamped_sources = list(sources)
        self._stamped_thumb_width = thumb_width
        log.info("Stacker.stamp.completed")

        return stamps_list

    def stack(
        self,
        sources: list[RegisteredSource],
        method: StackingMethod,
        thumb_width=20 * u.arcmin,
    ) -> list[list[StackedStamp]]:
        log = self.log.bind(func="Stacker.stack", method=type(method).__name__)
        cached = (
            self.stamps is not None
            and self._stamped_sources == sources
            and self._stamped_thumb_width == thumb_width
        )
        log.debug(
            "Stacker.stack.using_cached_stamps"
            if cached
            else "Stacker.stack.restamping"
        )
        stamps = self.stamps if cached else self.stamp(sources, thumb_width)

        return [
            method.stack(s, log=log.bind(source_id=src.source_id))
            for src, s in zip(sources, stamps)
        ]
