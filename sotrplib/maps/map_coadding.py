from abc import ABC, abstractmethod

import astropy.units as u
import numpy as np
import structlog
from pixell import enmap
from pixell.enmap import ndmap
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import CoaddedRhoKappaMap, ProcessableMap


class MapCoadder(ABC):
    """
    Docstring for MapCoadder
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
    input_maps : list of ProcessableMap
        The list of input maps to be coadded. Each should be a `RhoAndKappaMap`.
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
        self.map_depth = None
        self.frequencies = frequencies
        self.arrays = arrays if arrays is not None else ["coadd"]
        self.instrument = instrument

        self.input_map_times: list[float] = []
        self.time_last = None
        self.time_first = None
        self.time_mean = None
        self.observation_start = None
        self.observation_end = None
        self.map_depth = None
        self.log = log or structlog.get_logger()

    def coadd(self, input_maps: list[ProcessableMap]):
        """
        Coadd input_maps given the coadder freqs, arrays.

        Make coadd for each arr, freq in self.arrays, self.frequencies

        if self.arrays=None just make one coadded map over all arrays, called "coadd".

        if self.frequencies=None, get unique frequencies from input maps.

        returns a list of Coadded ProcessableMap
        """
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

                self.time_last = None
                self.time_first = None
                self.time_mean = None
                self.observation_start = None
                self.observation_end = None
                self.map_depth = None

                self.log.info(
                    "rhokappamapcoadder.base_map.built",
                    map_start_time=base_map.observation_start,
                    map_end_time=base_map.observation_end,
                    frequency=freq,
                    array=arr,
                )

                self.rho = enmap.enmap(base_map.rho)
                self.kappa = enmap.enmap(base_map.kappa)
                self.mask = (
                    enmap.enmap(base_map.mask) if base_map.mask is not None else None
                )

                self.map_resolution = u.Quantity(
                    abs(base_map.rho.wcs.wcs.cdelt[0]), base_map.rho.wcs.wcs.cunit[0]
                )
                self.flux_units = base_map.flux_units

                self.get_time_and_mapdepth(base_map)
                self.update_map_times(base_map)

                if len(valid_input_maps) == 1:
                    self.log.warning(
                        "rhokappamapcoadder.coadd.single_map_warning", n_maps_coadded=1
                    )
                    self.n_maps = 1
                    coadded_maps.append(base_map)
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
                    if sourcemap.flux_units != self.flux_units:
                        flux_conv = u.Quantity(1.0, sourcemap.flux_units).to(
                            self.flux_units
                        )
                        sourcemap.rho *= flux_conv
                        sourcemap.kappa /= flux_conv * flux_conv

                    self.rho = enmap.map_union(
                        self.rho,
                        sourcemap.rho,
                    )
                    self.kappa = enmap.map_union(
                        self.kappa,
                        sourcemap.kappa,
                    )
                    if self.mask is not None and sourcemap.mask is not None:
                        self.mask = enmap.map_union(
                            self.mask,
                            enmap.enmap(sourcemap.mask),
                        )
                    elif sourcemap.mask is not None:
                        self.mask = enmap.enmap(sourcemap.mask)

                    self.get_time_and_mapdepth(sourcemap)
                    self.update_map_times(sourcemap)

                self.n_maps = len(self.input_map_times)
                self.log.info(
                    "rhokappamapcoadder.coadd.finalized",
                    n_maps_coadded=self.n_maps,
                    coadd_start_time=self.observation_start,
                    coadd_end_time=self.observation_end,
                    freq=freq,
                    arr=arr,
                )
                self.observation_length = self.observation_end - self.observation_start
                with np.errstate(divide="ignore"):
                    self.time_mean /= self.map_depth

                self.mask[self.mask > 0] = 1

                coadded_maps.append(
                    CoaddedRhoKappaMap(
                        rho=self.rho,
                        kappa=self.kappa,
                        observation_start=self.observation_start,
                        observation_end=self.observation_end,
                        time_first=self.time_first,
                        time_mean=self.time_mean,
                        time_last=self.time_last,
                        observation_length=self.observation_length,
                        array=arr,
                        frequency=freq,
                        instrument=self.instrument,
                        flux_units=self.flux_units,
                        mask=self.mask,
                        map_resolution=self.map_resolution,
                        hits=self.map_depth,
                    )
                )

        return coadded_maps

    def get_time_and_mapdepth(self, new_map):
        if isinstance(new_map.time_mean, ndmap):
            if self.time_mean is None:
                self.time_mean = enmap.enmap(new_map.time_mean)
                self.map_depth = enmap.enmap(new_map.time_mean)
                self.map_depth[self.map_depth > 0] = 1.0
            else:
                self.time_mean = enmap.map_union(
                    self.time_mean,
                    new_map.time_mean,
                )
                new_map_depth = enmap.enmap(new_map.time_mean)
                new_map_depth[new_map_depth > 0] = 1.0
                self.map_depth = enmap.map_union(
                    self.map_depth,
                    new_map_depth,
                )
        else:
            self.log.error(
                "rhokappamapcoadder.get_time_and_mapdepth.no_time_mean",
            )
        ## get earliest start time per pixel ... can I use map_union with a<b or b<a or something?
        if isinstance(new_map.time_first, ndmap):
            if self.time_first is None:
                self.time_first = enmap.enmap(new_map.time_first)
            else:
                self.time_first = pixell_map_union(
                    self.time_first,
                    new_map.time_first,
                    op=lambda a, b: np.minimum(a, b),
                )
        else:
            self.log.error(
                "rhokappamapcoadder.get_time_and_mapdepth.no_time_first",
            )
        ## get latest end time per pixel ... can I use map_union with a>b or b>a or something?
        if isinstance(new_map.time_last, ndmap):
            if self.time_last is None:
                self.time_last = enmap.enmap(new_map.time_last)
            else:
                self.time_last = pixell_map_union(
                    self.time_last,
                    new_map.time_last,
                    op=lambda a, b: np.maximum(a, b),
                )
        else:
            self.log.error(
                "rhokappamapcoadder.get_time_and_mapdepth.no_time_last",
            )

    def update_map_times(self, new_map):
        if self.observation_start is None:
            self.observation_start = new_map.observation_start
        else:
            self.observation_start = min(
                self.observation_start, new_map.observation_start
            )
        if self.observation_end is None:
            self.observation_end = new_map.observation_end
        else:
            self.observation_end = max(self.observation_end, new_map.observation_end)
        time_delta = new_map.observation_end - new_map.observation_start
        mid_time = new_map.observation_start + (time_delta / 2)
        self.input_map_times.append(mid_time)


def pixell_map_union(map1, map2, op=lambda a, b: a + b):
    """Create a new pixell map that is the union of map1 and map2.
    The new map will have the shape and wcs that covers both input maps.
    The pixel values will be combined using the provided operation.

    Args:
        map1: First input pixell map.
        map2: Second input pixell map.
        op: Function to combine pixel values (default is addition).

    """
    from pixell import enmap

    oshape, owcs = enmap.union_geometry([map1.geometry, map2.geometry])
    omap = enmap.zeros(map1.shape[:-2] + oshape[-2:], owcs, map1.dtype)
    omap.insert(map1)
    omap.insert(map2, op=op)
    return omap
