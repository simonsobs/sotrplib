import re
from abc import ABC, abstractmethod

import astropy.units as u
import numpy as np
import structlog
from pixell import enmap
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import (
    CoaddedIntensityAndInverseVarianceMap,
    CoaddedRhoKappaMap,
    ProcessableMap,
)

_ARRAY_TOKEN_RE = re.compile(r"[a-zA-Z]+\d+")


def _combined_array_label(input_maps: list[ProcessableMap]) -> str | None:
    """
    Label for a coadd's `array` attribute, reflecting which array(s) actually
    went into it. A single array's maps keep that array's name (e.g. "i6");
    maps from several arrays combined together (e.g. via arrays=["coadd"])
    get a concatenated label of each *unique* array (e.g. "i1i3i6").

    Re-parses each input map's `.array` string into its atomic tokens (e.g.
    "i1", "c1") rather than treating the whole string as one opaque unit.
    This matters for streaming/incremental coadding, where coadd_maps() is
    called repeatedly as [running_coadd, new_map]: running_coadd.array is
    itself already a combined multi-array label from the previous merge, so
    treating it as a single token would keep re-concatenating the same
    tokens on every merge instead of deduplicating against them.
    """
    tokens: set[str] = set()
    for m in input_maps:
        if m.array is not None:
            tokens.update(_ARRAY_TOKEN_RE.findall(m.array))
    if not tokens:
        return None
    return "".join(sorted(tokens))


class MapCoadder(ABC):
    """
    Abstract base class for map coadding strategies.
    Subclasses should implement the `coadd` method, which takes a list of
    `ProcessableMap` instances and returns a list of coadded `ProcessableMap` objects.
    This interface allows for different coadding algorithms to be implemented
    and used interchangeably.
    """

    @abstractmethod
    def coadd_maps(self, input_maps: list[ProcessableMap]) -> ProcessableMap:
        """Coadd a list of ProcessableMap objects into a single coadded map."""
        return

    @abstractmethod
    def group_maps(
        self, input_maps: list[ProcessableMap]
    ) -> list[list[ProcessableMap]]:
        """
        Group input maps by frequency and array. Returns a list of lists of
        ProcessableMap objects.
        """
        return

    @abstractmethod
    def coadd(self, input_maps: list[ProcessableMap]) -> list[ProcessableMap]:
        """
        Coadd all maps according the coadder's grouping strategy and return a list of
        coadded maps.
        """
        return


class EmptyMapCoadder(MapCoadder):
    def coadd_maps(self, input_maps: list[ProcessableMap]) -> ProcessableMap:
        assert len(input_maps) == 1, "EmptyMapCoadder can only coadd a single map"
        return input_maps[0]

    def group_maps(self, input_maps):
        return [[input_map] for input_map in input_maps]

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

    def group_maps(
        self, input_maps: list[ProcessableMap]
    ) -> list[list[ProcessableMap]]:
        """
        Group input maps by frequency and array. Returns a list of lists of maps.
        Within each list, the maps all have the same frequency and array.
        """
        map_sets = list()
        for arr in self.arrays:
            for freq in self.frequencies:
                map_sets.append(self._get_valid_maps(input_maps, freq, arr))
        map_sets = [ms for ms in map_sets if ms]  # filter out empty map sets
        return map_sets

    def _get_valid_maps(self, input_maps: list[ProcessableMap], freq: str, arr: str):
        """
        Identify which maps match the provided frequency and array.
        """
        good_maps = [True] * len(input_maps)
        for i, imap in enumerate(input_maps):
            if imap.frequency != freq:
                good_maps[i] = False
            if arr != "coadd" and arr != imap.array:
                good_maps[i] = False

        if not any(good_maps):
            self.log.debug(
                "rhokappamapcoadder.coadd.no_good_maps",
                n_input_maps=len(input_maps),
                frequency=freq,
                array=arr,
            )
            return list()
        if not all(good_maps):
            self.log.info(
                "rhokappamapcoadder.coadd.dropping_maps",
                n_input_maps=len(input_maps),
                n_dropped_maps=len([good for good in good_maps if not good]),
                frequency=freq,
                array=arr,
            )

        valid_input_maps = [imap for imap, good in zip(input_maps, good_maps) if good]

        return valid_input_maps

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
                valid_input_maps = self._get_valid_maps(input_maps, freq, arr)

                if not valid_input_maps:
                    continue

                coadded_maps.append(self.coadd_maps(valid_input_maps))

        return coadded_maps

    def coadd_maps(self, valid_input_maps: list[ProcessableMap]) -> ProcessableMap:
        base_map = valid_input_maps[0]
        base_map.build()
        freq = base_map.frequency
        arr = _combined_array_label(valid_input_maps)

        coadd = CoaddedRhoKappaMap(
            rho=base_map.rho,
            kappa=base_map.kappa,
            observation_start=base_map.observation_start,
            observation_end=base_map.observation_end,
            time_first=base_map.time_first,
            time_mean=base_map.time_mean,
            time_last=base_map.time_last,
            observation_length=base_map.observation_end - base_map.observation_start,
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
            return coadd

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
                flux_conv = u.Quantity(1.0, sourcemap.flux_units).to(coadd.flux_units)
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
            coadd._hits = enmap.map_union(
                coadd.hits,
                sourcemap.hits,
            )
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
        self.log.info(
            "rhokappamapcoadder.sourcemap.time_mean",
            time_mean=sourcemap.time_mean,
            alt_time_mean=coadd.time_mean,
        )
        coadd.observation_length = coadd.observation_end - coadd.observation_start

        if coadd.mask is not None:
            coadd.mask[coadd.mask > 0] = 1

        return coadd


class IntensityMapCoadder(MapCoadder):
    """
    Coadd raw (unfiltered) intensity/inverse-variance depth-1 maps into an
    inverse-variance-weighted mean map, per frequency/array.

    Unlike RhoKappaMapCoadder, the inputs are not required to have been
    matched filtered already; matched filtering (or other preprocessing)
    should instead be applied to the resulting coadd, e.g. via the pipeline's
    `preprocessors` config, matching the "coadd then preprocess" workflow in
    docs/flowchart.md.

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

    def group_maps(
        self, input_maps: list[ProcessableMap]
    ) -> list[list[ProcessableMap]]:
        """
        Group input maps by frequency and array. Returns a list of lists of maps.
        Within each list, the maps all have the same frequency and array.
        """
        map_sets = list()
        for arr in self.arrays:
            for freq in self.frequencies:
                map_sets.append(self._get_valid_maps(input_maps, freq, arr))
        map_sets = [ms for ms in map_sets if ms]  # filter out empty map sets
        return map_sets

    def _get_valid_maps(self, input_maps: list[ProcessableMap], freq: str, arr: str):
        """
        Identify which maps match the provided frequency and array.
        """
        good_maps = [True] * len(input_maps)
        for i, imap in enumerate(input_maps):
            if imap.frequency != freq:
                good_maps[i] = False
            if arr != "coadd" and arr != imap.array:
                good_maps[i] = False

        if not any(good_maps):
            self.log.debug(
                "intensitymapcoadder.coadd.no_good_maps",
                n_input_maps=len(input_maps),
                frequency=freq,
                array=arr,
            )
            return list()
        if not all(good_maps):
            self.log.info(
                "intensitymapcoadder.coadd.dropping_maps",
                n_input_maps=len(input_maps),
                n_dropped_maps=len([good for good in good_maps if not good]),
                frequency=freq,
                array=arr,
            )

        valid_input_maps = [imap for imap, good in zip(input_maps, good_maps) if good]

        return valid_input_maps

    def coadd(self, input_maps: list[ProcessableMap]):
        """
        Coadd input_maps given the coadder freqs, arrays.

        Make coadd for each arr, freq in self.arrays, self.frequencies

        if self.arrays=None just make one coadded map over all arrays, called "coadd".

        if self.frequencies=None, get unique frequencies from input maps.

        returns a list of Coadded ProcessableMap
        """
        if not input_maps:
            self.log.warning("intensitymapcoadder.coadd.no_input_maps")
            return []

        if self.frequencies is None:
            self.frequencies = sorted(list(set(imap.frequency for imap in input_maps)))
            self.log.info(
                "intensitymapcoadder.coadd.getting_freqs",
                frequencies=self.frequencies,
            )

        coadded_maps = []
        for arr in self.arrays:
            for freq in self.frequencies:
                valid_input_maps = self._get_valid_maps(input_maps, freq, arr)

                if not valid_input_maps:
                    continue

                coadded_maps.append(self.coadd_maps(valid_input_maps))

        return coadded_maps

    def coadd_maps(
        self, valid_input_maps: list[ProcessableMap]
    ) -> CoaddedIntensityAndInverseVarianceMap:
        base_map = valid_input_maps[0]
        base_map.build()
        freq = base_map.frequency
        arr = _combined_array_label(valid_input_maps)

        coadd = CoaddedIntensityAndInverseVarianceMap(
            intensity=base_map.intensity * base_map.inverse_variance,
            inverse_variance=base_map.inverse_variance,
            observation_start=base_map.observation_start,
            observation_end=base_map.observation_end,
            time_first=base_map.time_first,
            time_mean=base_map.time_mean,
            time_last=base_map.time_last,
            observation_length=base_map.observation_end - base_map.observation_start,
            array=arr,
            frequency=freq,
            instrument=self.instrument,
            intensity_units=base_map.intensity_units,
            mask=base_map.mask,
            map_resolution=base_map.map_resolution,
            hits=base_map.hits,
            map_ids=[base_map.map_id],
        )

        self.log.info(
            "intensitymapcoadder.coadd.built",
            map_start_time=coadd.observation_start,
            map_end_time=coadd.observation_end,
            frequency=freq,
            array=arr,
        )

        if len(valid_input_maps) > 1:
            for sourcemap in valid_input_maps[1:]:
                sourcemap.build()
                self.log.info(
                    "intensitymapcoadder.source_map.built",
                    map_start_time=sourcemap.observation_start,
                    map_end_time=sourcemap.observation_end,
                    map_frequency=sourcemap.frequency,
                )
                sourcemap_intensity = sourcemap.intensity
                sourcemap_ivar = sourcemap.inverse_variance
                if sourcemap.intensity_units != coadd.intensity_units:
                    conv = u.Quantity(1.0, sourcemap.intensity_units).to(
                        coadd.intensity_units
                    )
                    sourcemap_intensity = sourcemap_intensity * conv
                    sourcemap_ivar = sourcemap_ivar / (conv * conv)

                coadd.intensity = enmap.map_union(
                    coadd.intensity,
                    sourcemap_intensity * sourcemap_ivar,
                )
                coadd.inverse_variance = enmap.map_union(
                    coadd.inverse_variance,
                    sourcemap_ivar,
                )
                if coadd.mask is not None and sourcemap.mask is not None:
                    coadd.mask = enmap.map_union(
                        coadd.mask,
                        enmap.enmap(sourcemap.mask),
                    )
                elif sourcemap.mask is not None:
                    coadd.mask = enmap.enmap(sourcemap.mask)

                coadd.update_times(sourcemap)
                coadd._hits = enmap.map_union(
                    coadd.hits,
                    sourcemap.hits,
                )
                coadd.map_ids.append(sourcemap.map_id)
        else:
            self.log.warning(
                "intensitymapcoadder.coadd.single_map_warning", n_maps_coadded=1
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            coadded_intensity = coadd.intensity / coadd.inverse_variance
        coadded_intensity[coadd.inverse_variance <= 0] = 0.0
        coadd.intensity = coadded_intensity

        n_maps = len(coadd.input_map_times) if coadd.input_map_times else 1
        self.log.info(
            "intensitymapcoadder.coadd.completed",
            n_maps_coadded=n_maps,
            coadd_start_time=coadd.observation_start,
            coadd_end_time=coadd.observation_end,
            freq=freq,
            arr=arr,
        )
        coadd.observation_length = coadd.observation_end - coadd.observation_start

        if coadd.mask is not None:
            coadd.mask[coadd.mask > 0] = 1

        return coadd
