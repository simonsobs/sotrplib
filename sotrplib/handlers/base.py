from typing import Iterable

import numpy as np
import structlog
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
)
from sotrplib.maps.postprocessor import MapPostprocessor
from sotrplib.maps.preprocessor import MapPreprocessor
from sotrplib.maps.utils import enmap_box_to_skycoord
from sotrplib.outputs.core import MapOutput, SourceOutput
from sotrplib.sifter.core import EmptySifter, SifterResult, SiftingProvider
from sotrplib.sifter.map_matching import EmptyMapMatcher, MapMatcher, MapResult
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
    map_matcher: MapMatcher | None
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
        map_matcher: MapMatcher | None = None,
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
        self.map_matcher = map_matcher or EmptyMapMatcher()
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
                self.profilable_task(set_processing_end)(
                    input_map.mapcat_id, map_type=input_map.map_type
                )
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

        start_time = None
        end_time = None
        for input_map in maps:
            if start_time is None or input_map.observation_start < start_time:
                start_time = input_map.observation_start
            if end_time is None or input_map.observation_end > end_time:
                end_time = input_map.observation_end

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
    ) -> MapResult | None:
        """
        Coadd and analyze maps in a single task to avoid passing maps between processes.
        """
        coadded_map = self.profilable_task(self.map_coadder.coadd_maps)(maps)
        return self.profilable_task(self.analyze_map)(
            input_map=coadded_map, simulated_sources=simulated_sources
        )

    def analyze_map(
        self, input_map: ProcessableMap, simulated_sources: list[SimulatedSource]
    ) -> MapResult | None:
        """
        Analyze one map and write its map outputs. Source outputs are not
        written here: they wait for map matching, which needs every map's
        results (see output_map_result). Returns None for an empty map.
        """
        # If an exception occurs, set the processing status to failed.
        try:
            input_map = self.profilable_task(self.build_map)(input_map)

            if input_map is not None:
                self.profilable_task(input_map.finalize)()
            else:
                return None

            injected_sources, input_map = self.profilable_task(
                self.source_injector.inject
            )(input_map=input_map, simulated_sources=simulated_sources)

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
                # The key of the pointing-residual table is the depth-1
                # map_id. Thus, do not save the model of a coadd.
                if (
                    input_map._parent_database is not None
                    and input_map.map_type == "depth1_map"
                ):
                    save_pointing_model(
                        input_map.mapcat_id, pointing_model, pointing_model_stats
                    )

            forced_photometry_candidates = self.profilable_task(
                self.forced_photometry.force
            )(
                input_map=input_map,
                catalogs=self.source_catalogs,
                pointing_model=pointing_model,
            )

            source_subtracted_map = self.profilable_task(
                self.source_subtractor.subtract
            )(sources=forced_photometry_candidates, input_map=input_map)

            blind_sources, _ = self.profilable_task(self.blind_search.search)(
                input_map=source_subtracted_map,
                pointing_model=pointing_model,
            )

            sifter_result = self.profilable_task(self.sifter.sift)(
                sources=blind_sources,
                catalogs=self.source_catalogs,
                input_map=source_subtracted_map,
            )

            for output in self.map_outputs:
                self.profilable_task(output.output)(input_map=input_map)

            return MapResult(
                map_name=input_map.map_name,
                mapcat_id=input_map.mapcat_id,
                map_type=input_map.map_type,
                array=input_map.array,
                frequency=input_map.frequency,
                observation_start=input_map.observation_start,
                forced_photometry_candidates=forced_photometry_candidates,
                sifter_result=sifter_result,
                pointing_sources=pointing_sources,
                injected_sources=injected_sources,
                from_database=input_map._parent_database is not None,
            )
        except Exception:
            # Set the status to "failed". If the status stays "processing",
            # the reader skips the map on the next run.
            if getattr(input_map, "_parent_database", None) is not None:
                self.profilable_task(set_processing_end)(
                    input_map.mapcat_id, map_type=input_map.map_type, status="failed"
                )
            raise

    def output_map_result(self, result: MapResult) -> None:
        """
        Write one map's source outputs, after map matching, and only then
        mark the map completed.
        """
        try:
            for output in self.source_outputs:
                self.profilable_task(output.output)(
                    forced_photometry_candidates=result.forced_photometry_candidates,
                    sifter_result=result.sifter_result,
                    map_name=result.map_name,
                    mapcat_id=result.mapcat_id,
                    pointing_sources=result.pointing_sources,
                    injected_sources=result.injected_sources,
                )
        except Exception:
            if result.from_database:
                self.profilable_task(set_processing_end)(
                    result.mapcat_id, map_type=result.map_type, status="failed"
                )
            raise

        if result.from_database:
            self.profilable_task(set_processing_end)(
                result.mapcat_id, map_type=result.map_type
            )

    def run(
        self, maps: list[ProcessableMap]
    ) -> list[tuple[list[MeasuredSource], SifterResult]]:
        return self.flow(self._run)(maps)

    def _run(
        self, maps: list[ProcessableMap]
    ) -> list[tuple[list[MeasuredSource], SifterResult]]:
        """
        The actual pipeline run logic has to be in a separate method so that it can be
        decorated with the flow as prefect needs these to be defined in advance.

        Returns one (forced photometry sources, sifter result) tuple for each
        map set. If there are no maps, returns an empty list.
        """
        maps = list(maps)
        if not maps:
            structlog.get_logger().warning(
                "pipeline.no_maps_found",
                message="No input maps. Check the map configuration or database query.",
            )
            return []
        sky_box = self.extract_bounding_box(maps)
        time_range = self.observation_time_range(maps)
        all_simulated_sources = self.basic_task(self.simulate_sources)(
            sky_box, time_range
        )
        map_sets = self.basic_task(self.map_coadder.group_maps)(maps)
        map_results = (
            self.basic_task(self.coadd_and_analyze_maps)
            .map(map_sets, self.unmapped(all_simulated_sources))
            .result()
        )
        # Held in memory until every map is analyzed, so map matching can
        # group transient candidates across maps before anything is written.
        map_results = [r for r in map_results if r is not None]
        map_results, _ = self.profilable_task(self.map_matcher.match)(map_results)

        self.basic_task(self.output_map_result).map(map_results).result()

        return [(r.forced_photometry_candidates, r.sifter_result) for r in map_results]
