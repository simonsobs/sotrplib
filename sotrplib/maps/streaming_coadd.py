"""
Memory-bounded coadding: build, preprocess (e.g. matched filter), and merge
one depth-1 map at a time, discarding each raw map before moving on to the
next, rather than holding every input map in memory at once.

Unlike the main pipeline (see docs/flowchart.md), which coadds first and
preprocesses the result once, this applies the preprocessor chain to each
*individual* depth-1 map before it is folded into the running coadd. That
matters for two reasons: moving sources (asteroids, satellites, ...) smear
across pixels if many days of raw maps are summed before any per-observation
handling, and matched filtering needs each observation's own noise
properties rather than those of an already-blurred sum.
"""

from typing import Iterable

from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.map_coadding import RhoKappaMapCoadder
from sotrplib.maps.preprocessor import MapPreprocessor


def stream_coadd(
    maps: Iterable[ProcessableMap],
    preprocessors: list[MapPreprocessor],
    coadder: RhoKappaMapCoadder,
    log: FilteringBoundLogger | None = None,
) -> tuple[ProcessableMap | None, list[str]]:
    """
    Build, preprocess, and merge `maps` into a single coadd one at a time.

    Parameters
    ----------
    maps : Iterable[ProcessableMap]
        Unbuilt input maps (e.g. from a MapCatDatabaseReader). Only one
        map's pixel data is held in memory at a time.
    preprocessors : list[MapPreprocessor]
        Applied, in order, to each individual map before it is merged in
        (e.g. planet masking, matched filtering, kappa/rho cleaning, edge
        masking) -- the same objects used by the main pipeline's
        `preprocessors` config, just run per map instead of once on the
        final coadd.
    coadder : RhoKappaMapCoadder
        Used to merge each preprocessed map into the running coadd via
        repeated `coadd_maps([running, filtered])` calls, which is a
        correct incremental merge: it builds a new CoaddedRhoKappaMap each
        call rather than mutating `running` in place.

    Returns
    -------
    (coadd, map_ids) : The final coadd (None if `maps` was empty) and the
        list of every input map's `map_id`, in the order merged. Tracked
        here rather than read off `coadd.map_ids` afterwards, because
        `coadd_maps()` seeds `map_ids` from `base_map.map_id` (singular) --
        when `base_map` is itself a running coadd from a previous
        iteration, that drops everything merged before it.
    """
    log = log or get_logger()

    running: ProcessableMap | None = None
    map_ids: list[str] = []

    for raw_map in maps:
        raw_map.build()
        map_id = raw_map.map_id

        filtered = raw_map
        for preprocessor in preprocessors:
            filtered = preprocessor.preprocess(input_map=filtered)

        running = (
            coadder.coadd_maps([filtered])
            if running is None
            else coadder.coadd_maps([running, filtered])
        )
        map_ids.append(map_id)

        log.info(
            "stream_coadd.merged_map",
            map_id=map_id,
            n_merged=len(map_ids),
        )

        del raw_map, filtered

    return running, map_ids
