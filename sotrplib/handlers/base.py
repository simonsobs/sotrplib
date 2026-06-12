from datetime import datetime, timezone
from itertools import combinations
from typing import Iterable

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from mapcat.pointing.const import ConstantPointingModel
from mapcat.pointing.poly import PolynomialPointingModel

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.database import save_pointing_model, set_processing_end
from sotrplib.maps.map_coadding import EmptyMapCoadder, MapCoadder
from sotrplib.maps.pointing import (
    EmptyPointingOffset,
    MapPointingOffset,
    save_model_maps,
)
from sotrplib.maps.postprocessor import MapPostprocessor
from sotrplib.maps.preprocessor import MapPreprocessor
from sotrplib.maps.utils import enmap_box_to_skycoord
from sotrplib.outputs.core import MapOutput, SourceOutput
from sotrplib.sifter.core import EmptySifter, SifterResult, SiftingProvider
from sotrplib.sifter.crossmatch import crossmatch_mask, n_wise_crossmatch
from sotrplib.sims.sim_source_generators import (
    SimulatedSource,
    SimulatedSourceGenerator,
)
from sotrplib.sims.source_injector import EmptySourceInjector, SourceInjector
from sotrplib.source_catalog.core import SourceCatalog
from sotrplib.sources.blind import EmptyBlindSearch
from sotrplib.sources.core import (
    BlindSearchProvider,
    ForcedPhotometryProvider,
)
from sotrplib.sources.force import EmptyForcedPhotometry
from sotrplib.sources.sources import MeasuredSource
from sotrplib.sources.subtractor import EmptySourceSubtractor, SourceSubtractor

__all__ = ["BaseRunner"]


class BaseRunner:
    maps: Iterable[ProcessableMap]
    map_coadder: MapCoadder | None
    source_simulators: list[SimulatedSourceGenerator] | None
    source_injector: SourceInjector | None
    source_catalogs: list[SourceCatalog] | None
    preprocessors: list[MapPreprocessor] | None
    pointing_provider: ForcedPhotometryProvider | None
    pointing_residual_model: MapPointingOffset | None
    postprocessors: list[MapPostprocessor] | None
    forced_photometry: ForcedPhotometryProvider | None
    source_subtractor: SourceSubtractor | None
    blind_search: BlindSearchProvider | None
    sifter: SiftingProvider | None
    source_outputs: list[SourceOutput] | None
    map_outputs: list[MapOutput] | None
    profile: bool = False

    def __init__(
        self,
        map_coadder: MapCoadder | None,
        source_simulators: list[SimulatedSourceGenerator] | None,
        source_injector: SourceInjector | None,
        source_catalogs: list[SourceCatalog] | None,
        preprocessors: list[MapPreprocessor] | None,
        pointing_provider: ForcedPhotometryProvider | None,
        pointing_residual_model: MapPointingOffset | None,
        postprocessors: list[MapPostprocessor] | None,
        forced_photometry: ForcedPhotometryProvider | None,
        source_subtractor: SourceSubtractor | None,
        blind_search: BlindSearchProvider | None,
        sifter: SiftingProvider | None,
        source_outputs: list[SourceOutput] | None,
        map_outputs: list[MapOutput] | None,
        profile: bool = False,
    ):
        self.map_coadder = map_coadder or EmptyMapCoadder()
        self.source_simulators = source_simulators or []
        self.source_injector = source_injector or EmptySourceInjector()
        self.source_catalogs = source_catalogs or []
        self.preprocessors = preprocessors or []
        self.pointing_provider = pointing_provider or EmptyForcedPhotometry()
        self.pointing_residual_model = pointing_residual_model or EmptyPointingOffset()
        self.postprocessors = postprocessors or []
        self.forced_photometry = forced_photometry or EmptyForcedPhotometry()
        self.source_subtractor = source_subtractor or EmptySourceSubtractor()
        self.blind_search = blind_search or EmptyBlindSearch()
        self.sifter = sifter or EmptySifter()
        self.source_outputs = source_outputs or []
        self.map_outputs = map_outputs or []
        self.profile = profile

    @property
    def profilable_task(self):
        raise NotImplementedError

    @property
    def basic_task(self):
        raise NotImplementedError

    @property
    def flow(self):
        raise NotImplementedError

    @property
    def unmapped(self):
        raise NotImplementedError

    def build_map(self, input_map: ProcessableMap) -> ProcessableMap:
        self.profilable_task(input_map.build)()
        output_map = input_map
        if not np.any(output_map.hits > 0):
            if input_map._parent_database is not None:
                set_processing_end(input_map.map_id)
            return None
        for preprocessor in self.preprocessors:
            output_map = self.profilable_task(preprocessor.preprocess)(
                input_map=output_map
            )

        return output_map

    def coadd_maps(self, input_maps: list[ProcessableMap]) -> list[ProcessableMap]:
        return self.map_coadder.coadd(input_maps)

    def extract_bounding_box(
        self, maps: list[ProcessableMap] | None = None
    ) -> tuple[SkyCoord, SkyCoord] | None:
        if not maps:
            return None

        # set defaults so that any real map will update them.
        dec_min = np.inf
        dec_max = -np.inf
        ra_min = np.inf
        ra_max = -np.inf
        for input_map in maps:
            b = input_map.bbox  # map.bbox returns a pixell box [[dec_min, ra_max], [dec_max, ra_min]] in radians
            dec_min = min(dec_min, b[0][0], b[1][0])
            dec_max = max(dec_max, b[0][0], b[1][0])
            ra_min = min(ra_min, b[0][1], b[1][1])
            ra_max = max(ra_max, b[0][1], b[1][1])
        bbox = np.array([[dec_min, ra_max], [dec_max, ra_min]])
        sky_box = enmap_box_to_skycoord(bbox)
        return sky_box

    def observation_time_range(self, maps=None) -> tuple[Time, Time]:
        """
        Get the time range covering all input maps.
        """
        if not maps:
            return (None, None)
        start_time = datetime.max.replace(tzinfo=timezone.utc)
        end_time = datetime.min.replace(tzinfo=timezone.utc)
        for input_map in maps:
            start_time = min(input_map.observation_start, start_time)
            end_time = max(input_map.observation_end, end_time)
        return (start_time, end_time)

    def simulate_sources(
        self, sky_box: tuple[SkyCoord, SkyCoord] | None, time_range: tuple[float]
    ) -> list[SimulatedSource]:
        """Generate sources based upon maximal bounding box of all maps"""
        if len(self.source_simulators) == 0:
            return []
        all_simulated_sources = []
        for simulator in self.source_simulators:
            simulated_sources, catalog = self.profilable_task(simulator.generate)(
                sky_box=sky_box,
                time_range=time_range,
            )

            all_simulated_sources.extend(simulated_sources)
            self.source_catalogs.append(catalog)
        return all_simulated_sources

    def coadd_and_analyze_maps(
        self, maps: list[ProcessableMap], simulated_sources: list[SimulatedSource]
    ) -> tuple[list[MeasuredSource], SifterResult]:
        """
        Coadd and analyze maps in a single task to avoid passing maps between processes.
        """
        coadded_map = self.profilable_task(self.map_coadder.coadd_maps)(maps)
        return self.profilable_task(self.analyze_map)(
            input_map=coadded_map, simulated_sources=simulated_sources
        )

    def analyze_map(
        self, input_map: ProcessableMap, simulated_sources: list[SimulatedSource]
    ) -> tuple[list[MeasuredSource], SifterResult]:
        input_map = self.profilable_task(self.build_map)(input_map)

        if input_map is not None:
            self.profilable_task(input_map.finalize)()
        else:
            return [], None

        injected_sources, input_map = self.profilable_task(self.source_injector.inject)(
            input_map=input_map, simulated_sources=simulated_sources
        )

        for postprocessor in self.postprocessors:
            input_map = self.profilable_task(postprocessor.postprocess)(
                input_map=input_map
            )

        pointing_sources = self.profilable_task(self.pointing_provider.force)(
            input_map=input_map, catalogs=self.source_catalogs
        )

        cached = getattr(input_map, "pointing_model", None)
        if isinstance(cached, (ConstantPointingModel | PolynomialPointingModel)):
            pointing_model = cached
        else:
            pointing_model, pointing_model_stats = self.profilable_task(
                self.pointing_residual_model.build_model
            )(pointing_sources=pointing_sources)
            if input_map._parent_database is not None:
                save_pointing_model(
                    input_map.map_id, pointing_model, pointing_model_stats
                )

        ## this is dumb and should be fixed.
        for o in self.map_outputs:
            if "pointing_residual_map" in o.field_ids:
                self.profilable_task(save_model_maps)(
                    self.pointing_residual_model,
                    pointing_model,
                    input_map,
                    filename_prefix=f"{o.directory}/pointing_residual_{input_map.map_id}",
                )
                break

        forced_photometry_candidates = self.profilable_task(
            self.forced_photometry.force
        )(
            input_map=input_map,
            catalogs=self.source_catalogs,
            pointing_model=pointing_model,
        )

        source_subtracted_map = self.profilable_task(self.source_subtractor.subtract)(
            sources=forced_photometry_candidates, input_map=input_map
        )

        blind_sources, _ = self.profilable_task(self.blind_search.search)(
            input_map=source_subtracted_map,
            pointing_model=pointing_model,
        )

        sifter_result = self.profilable_task(self.sifter.sift)(
            sources=blind_sources,
            catalogs=self.source_catalogs,
            input_map=source_subtracted_map,
        )

        for output in self.source_outputs:
            self.profilable_task(output.output)(
                forced_photometry_candidates=forced_photometry_candidates,
                sifter_result=sifter_result,
                map_id=input_map.map_id,
                pointing_sources=pointing_sources,
                injected_sources=injected_sources,
            )

        for output in self.map_outputs:
            self.profilable_task(output.output)(input_map=input_map)

        if input_map._parent_database is not None:
            self.profilable_task(set_processing_end)(input_map.map_id)
        return forced_photometry_candidates, sifter_result

    def crossmatch_pair(
        self, candidates: tuple[list], radius: float = 1.5
    ) -> list[list[tuple]]:
        positions_1 = np.array([[src.dec.value, src.ra.value] for src in candidates[0]])
        positions_2 = np.array([[src.dec.value, src.ra.value] for src in candidates[1]])
        if len(positions_1) == 0 or len(positions_2) == 0:
            return []

        # FIXME: use a better radius
        found, matches = self.profilable_task(crossmatch_mask)(
            positions_1, positions_2, radius=radius, return_matches=True
        )
        return matches

    def run(self, maps: list[ProcessableMap]) -> tuple[list[list], list[object]]:
        return self.flow(self._run)(maps)

    def _initialize_socat_time_ranges(self, t_start: Time, t_end: Time) -> None:
        for catalog in self.source_catalogs:
            if hasattr(catalog, "set_time_range"):
                catalog.set_time_range(t_start, t_end)

    def _run(self, maps: list[ProcessableMap]) -> tuple[list[list], list[object]]:
        """
        The actual pipeline run logic has to be in a separate method so that it can be
        decorated with the flow as prefect needs these to be defined in advance.
        """
        sky_box = self.extract_bounding_box(maps)
        time_range = self.observation_time_range(maps)
        self._initialize_socat_time_ranges(time_range[0], time_range[1])
        all_simulated_sources = self.basic_task(self.simulate_sources)(
            sky_box, time_range
        )
        map_sets = self.basic_task(self.map_coadder.group_maps)(maps)
        results = (
            self.basic_task(self.coadd_and_analyze_maps)
            .map(map_sets, self.unmapped(all_simulated_sources))
            .result()
        )
        all_transient_candidates = [res[1].transient_candidates for res in results]
        matches = (
            self.basic_task(self.crossmatch_pair)
            .map(combinations(all_transient_candidates, 2))
            .result()
        )

        cross_matches = self.profilable_task(n_wise_crossmatch)(
            matches,
            dict(zip([mm[0].map_id for mm in map_sets], all_transient_candidates)),
        )

        return results
