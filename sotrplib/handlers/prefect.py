from pathlib import Path

from astropy.coordinates import SkyCoord
from prefect import flow, task

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.postprocessor import MapPostprocessor
from sotrplib.maps.preprocessor import MapPreprocessor
from sotrplib.outputs.core import SourceOutput
from sotrplib.sifter.core import EmptySifter, SiftingProvider
from sotrplib.sims.sim_source_generators import SimulatedSourceGenerator, SimulatedSource
from sotrplib.source_catalog.core import SourceCatalog
from sotrplib.sims.source_injector import EmptySourceInjector, SourceInjector
from sotrplib.sources.blind import EmptyBlindSearch
from sotrplib.sources.core import (
    BlindSearchProvider,
    ForcedPhotometryProvider,
)
from sotrplib.sources.force import EmptyForcedPhotometry
from sotrplib.sources.subtractor import EmptySourceSubtractor, SourceSubtractor


class PrefectRunner:
    maps: list[ProcessableMap]
    source_simulators: list[SimulatedSourceGenerator] | None
    source_injector: SourceInjector | None
    source_catalogs: list[SourceCatalog] | None
    preprocessors: list[MapPreprocessor] | None
    postprocessors: list[MapPostprocessor] | None
    forced_photometry: ForcedPhotometryProvider | None
    source_subtractor: SourceSubtractor | None
    blind_search: BlindSearchProvider | None
    sifter: SiftingProvider | None
    outputs: list[SourceOutput] | None

    def __init__(
        self,
        maps: list[ProcessableMap],
        source_simulators: list[SimulatedSourceGenerator] | None,
        source_injector: SourceInjector | None,
        source_catalogs: list[SourceCatalog] | None,
        preprocessors: list[MapPreprocessor] | None,
        postprocessors: list[MapPostprocessor] | None,
        forced_photometry: ForcedPhotometryProvider | None,
        source_subtractor: SourceSubtractor | None,
        blind_search: BlindSearchProvider | None,
        sifter: SiftingProvider | None,
        outputs: list[SourceOutput] | None,
    ):
        self.maps = maps
        self.source_simulators = source_simulators or []
        self.source_injector = source_injector or EmptySourceInjector()
        self.source_catalogs = source_catalogs or []
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []
        self.forced_photometry = forced_photometry or EmptyForcedPhotometry()
        self.source_subtractor = source_subtractor or EmptySourceSubtractor()
        self.blind_search = blind_search or EmptyBlindSearch()
        self.sifter = sifter or EmptySifter()
        self.outputs = outputs or []

        return

    @task
    def build_map(self, input_map: ProcessableMap):
        task(input_map.build)()

        for preprocessor in self.preprocessors:
            task(preprocessor.preprocess)(input_map=input_map)

    def build_maps(self):
        for input_map in self.maps:
            self.build_map(input_map)

    @property
    def bbox(self):
        bbox = self.maps[0].bbox

        for input_map in self.maps[1:]:
            map_bbox = input_map.bbox
            left = min(bbox[0].ra, map_bbox[0].ra)
            bottom = min(bbox[0].dec, map_bbox[0].dec)
            right = max(bbox[1].ra, map_bbox[1].ra)
            top = max(bbox[1].dec, map_bbox[1].dec)
            bbox = [SkyCoord(ra=left, dec=bottom), SkyCoord(ra=right, dec=top)]
        return bbox

    @task
    def simulate_sources(self) -> list[SimulatedSource]:
        """Generate sources based upon maximal bounding box of all maps"""
        all_simulated_sources = []
        for simulator in self.source_simulators:
            simulated_sources, catalog = task(simulator.generate)(box=self.bbox)

            all_simulated_sources.extend(simulated_sources)
            self.source_catalogs.append(catalog)
        return all_simulated_sources

    @task
    def analyze_map(self, input_map: ProcessableMap, simulated_sources: list[SimulatedSource]):
            task(input_map.finalize)()

            input_map = task(self.source_injector.inject)(
                input_map=input_map, simulated_sources=simulated_sources
            )

            for postprocessor in self.postprocessors:
                task(postprocessor.postprocess)(input_map=input_map)

            forced_photometry_candidates = task(self.forced_photometry.force)(
                input_map=input_map, catalogs=self.source_catalogs
            )

            source_subtracted_map = task(self.source_subtractor.subtract)(
                sources=forced_photometry_candidates, input_map=input_map
            )

            blind_sources, _ = task(self.blind_search.search)(
                input_map=source_subtracted_map
            )

            sifter_result = task(self.sifter.sift)(
                sources=blind_sources,
                catalogs=self.source_catalogs,
                input_map=source_subtracted_map,
            )

            for output in self.outputs:
                task(output.output)(
                    forced_photometry_candidates=forced_photometry_candidates,
                    sifter_result=sifter_result,
                    input_map=input_map,
                )

    def analyze_maps(self, all_simulated_sources: list[SimulatedSource]):
        for input_map in self.maps:
            self.analyze_map(input_map, all_simulated_sources)

    @flow
    def run(self):
        self.build_maps()
        all_simulated_sources = self.simulate_sources()
        self.analyze_maps(all_simulated_sources)


@flow
def analyze_from_configuration(config: Path | str):
    from sotrplib.config.config import Settings

    pipeline = Settings.from_file(config).to_prefect()
    return pipeline.run()
