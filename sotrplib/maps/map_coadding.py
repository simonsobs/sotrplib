from abc import ABC, abstractmethod

import astropy.units as u
import structlog
from pixell import enmap
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import CoaddedRhoKappaMap, ProcessableMap


class MapCoadder(ABC):
    """
    Abstract base class for map coadding strategies.
    Subclasses should implement the `coadd` method, which takes a list of
    `ProcessableMap` instances and returns a list of coadded `ProcessableMap` objects.
    This interface allows for different coadding algorithms to be implemented
    and used interchangeably.
    """

    @abstractmethod
    def coadd(self, input_maps: list[ProcessableMap]) -> list[ProcessableMap]:
        return


class EmptyMapCoadder(MapCoadder):
    def coadd(self, input_maps: list[ProcessableMap]) -> list[ProcessableMap]:
        return input_maps


class RhoKappaMapCoadder(MapCoadder):
    """
    Represents a coadded map created by combining multiple input `RhoAndKappaMap` (or similar) maps.
    This class is used to coadd (combine) a list of input maps

    Parameters
    ----------
    frequencies : list of str
        The frequency bands of the maps to coadd ["f090", "f150", "f220", etc.].
        Coadds are split by band.
    arrays : list of str, optional
        The arrays over which to coadd. Defaults to ["coadd"] if not specified.
        If not specified all arrays are coadded.
    log : FilteringBoundLogger, optional
        Logger for logging messages. If not provided, a default logger is used.
    """

    def __init__(
        self,
        frequencies: list[str] | None = None,
        arrays: list[str] | None = None,
        instrument: str | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.frequencies = frequencies
        self.arrays = arrays if arrays is not None else ["coadd"]
        self.instrument = instrument
        self.log = log or structlog.get_logger()

    def coadd(self, input_maps: list[ProcessableMap]):
        """
        Coadd input_maps given the coadder freqs, arrays.

        Make coadd for each arr, freq in self.arrays, self.frequencies

        if self.arrays=None just make one coadded map over all arrays, called "coadd".

        if self.frequencies=None, get unique frequencies from input maps.

        returns a list of Coadded ProcessableMap
        """
        if not input_maps:
            self.log.warning("rhokappamapcoadder.coadd.no_input_maps")
            return []

        if self.frequencies is None:
            self.frequencies = sorted(list(set(imap.frequency for imap in input_maps)))
            self.log.info(
                "rhokappamapcoadder.coadd.getting_freqs",
                frequencies=self.frequencies,
            )

        coadded_maps = []
        for arr in self.arrays:
            for freq in self.frequencies:
                good_maps = [True] * len(input_maps)
                for i, imap in enumerate(input_maps):
                    if imap.frequency != freq:
                        good_maps[i] = False
                    if arr != "coadd" and arr != imap.array:
                        good_maps[i] = False
                if not any(good_maps):
                    self.log.warning(
                        "rhokappamapcoadder.coadd.no_good_maps",
                        n_input_maps=len(input_maps),
                        frequency=freq,
                        array=arr,
                    )
                    continue
                if not all(good_maps):
                    self.log.info(
                        "rhokappamapcoadder.coadd.dropping_maps",
                        n_input_maps=len(input_maps),
                        n_dropped_maps=len([good for good in good_maps if not good]),
                        frequency=freq,
                        array=arr,
                    )

                valid_input_maps = [
                    imap for imap, good in zip(input_maps, good_maps) if good
                ]

                base_map = valid_input_maps[0]
                base_map.build()

                coadd = CoaddedRhoKappaMap(
                    rho=base_map.rho,
                    kappa=base_map.kappa,
                    observation_start=base_map.observation_start,
                    observation_end=base_map.observation_end,
                    time_first=base_map.time_first,
                    time_mean=base_map.time_mean,
                    time_last=base_map.time_last,
                    observation_length=base_map.observation_end
                    - base_map.observation_start,
                    array=arr,
                    frequency=freq,
                    instrument=self.instrument,
                    flux_units=base_map.flux_units,
                    mask=base_map.mask,
                    map_resolution=base_map.map_resolution,
                    hits=base_map.hits,
                    map_ids=[base_map.map_id],
                )

                self.log.info(
                    "rhokappamapcoadder.coadd.built",
                    map_start_time=coadd.observation_start,
                    map_end_time=coadd.observation_end,
                    frequency=freq,
                    array=arr,
                )

                if len(valid_input_maps) == 1:
                    self.log.warning(
                        "rhokappamapcoadder.coadd.single_map_warning", n_maps_coadded=1
                    )
                    n_maps = 1
                    coadded_maps.append(coadd)
                    continue

                for sourcemap in valid_input_maps[1:]:
                    sourcemap.build()
                    self.log.info(
                        "rhokappamapcoadder.source_map.built",
                        map_start_time=sourcemap.observation_start,
                        map_end_time=sourcemap.observation_end,
                        map_frequency=sourcemap.frequency,
                    )
                    ## will want to do a weighted sum using inverse variance.
                    if sourcemap.flux_units != coadd.flux_units:
                        flux_conv = u.Quantity(1.0, sourcemap.flux_units).to(
                            coadd.flux_units
                        )
                        sourcemap.rho *= flux_conv
                        sourcemap.kappa /= flux_conv * flux_conv

                    coadd.rho = enmap.map_union(
                        coadd.rho,
                        sourcemap.rho,
                    )
                    coadd.kappa = enmap.map_union(
                        coadd.kappa,
                        sourcemap.kappa,
                    )
                    if coadd.mask is not None and sourcemap.mask is not None:
                        coadd.mask = enmap.map_union(
                            coadd.mask,
                            enmap.enmap(sourcemap.mask),
                        )
                    elif sourcemap.mask is not None:
                        coadd.mask = enmap.enmap(sourcemap.mask)

                    coadd.update_times(sourcemap)
                    coadd._hits += sourcemap._compute_hits()
                    coadd.map_ids.append(sourcemap.map_id)

                n_maps = len(coadd.input_map_times)
                self.log.info(
                    "rhokappamapcoadder.coadd.completed",
                    n_maps_coadded=n_maps,
                    coadd_start_time=coadd.observation_start,
                    coadd_end_time=coadd.observation_end,
                    freq=freq,
                    arr=arr,
                )
                coadd.observation_length = (
                    coadd.observation_end - coadd.observation_start
                )

                if coadd.mask is not None:
                    coadd.mask[coadd.mask > 0] = 1

                coadded_maps.append(coadd)

        return coadded_maps
