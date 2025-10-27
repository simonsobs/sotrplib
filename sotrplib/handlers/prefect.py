from pathlib import Path

from prefect import flow, task

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.postprocessor import MapPostprocessor
from sotrplib.maps.preprocessor import MapPreprocessor
from sotrplib.outputs.core import SourceOutput
from sotrplib.sifter.core import EmptySifter, SiftingProvider
from sotrplib.sims.sources.core import SourceSimulation
from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.source_catalog.socat import SOCatEmptyCatalog, SOCatFITSCatalog
from sotrplib.sources.blind import EmptyBlindSearch
from sotrplib.sources.core import (
    BlindSearchProvider,
    ForcedPhotometryProvider,
)
from sotrplib.sources.force import EmptyForcedPhotometry
from sotrplib.sources.subtractor import EmptySourceSubtractor, SourceSubtractor


class PrefectRunner:
    maps: list[ProcessableMap]
    preprocessors: list[MapPreprocessor] | None
    postprocessors: list[MapPostprocessor] | None
    source_simulators: list[SourceSimulation] | None
    forced_photometry: ForcedPhotometryProvider | None
    source_subtractor: SourceSubtractor | None
    blind_search: BlindSearchProvider | None
    sifter: SiftingProvider | None
    outputs: list[SourceOutput] | None

    def __init__(
        self,
        maps: list[ProcessableMap],
        forced_photometry_catalog: SOCatFITSCatalog | None,
        source_catalogs: list[SOCatFITSCatalog] | None,
        preprocessors: list[MapPreprocessor] | None,
        postprocessors: list[MapPostprocessor] | None,
        source_simulators: list[SourceSimulation] | None,
        forced_photometry: ForcedPhotometryProvider | None,
        source_subtractor: SourceSubtractor | None,
        blind_search: BlindSearchProvider | None,
        sifter: SiftingProvider | None,
        outputs: list[SourceOutput] | None,
    ):
        self.maps = maps
        self.forced_photometry_catalog = (
            forced_photometry_catalog or SOCatEmptyCatalog()
        )
        self.source_catalogs = source_catalogs or []
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []
        self.source_simulators = source_simulators or []
        self.forced_photometry = forced_photometry or EmptyForcedPhotometry()
        self.source_subtractor = source_subtractor or EmptySourceSubtractor()
        self.blind_search = blind_search or EmptyBlindSearch()
        self.sifter = sifter or EmptySifter()
        self.outputs = outputs or []

        return

    @task
    def analyze_map(
        self, input_map: ProcessableMap
    ) -> tuple[list, object, ProcessableMap]:
        task(input_map.build)()

        for preprocessor in self.preprocessors:
            input_map = task(preprocessor.preprocess)(input_map=input_map)

        task(input_map.finalize)()

        for postprocessor in self.postprocessors:
            input_map = task(postprocessor.postprocess)(input_map=input_map)

        for simulator in self.source_simulators:
            input_map, additional_sources = task(simulator.simulate)(
                input_map=input_map
            )
            sim_catalog = RegisteredSourceCatalog(sources=additional_sources)
            self.source_catalogs.append(sim_catalog)

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
            input_map=source_subtracted_map,
            catalogs=self.source_catalogs,
        )

        for output in self.outputs:
            task(output.output)(
                forced_photometry_candidates=forced_photometry_candidates,
                sifter_result=sifter_result,
                input_map=input_map,
            )
        return forced_photometry_candidates, sifter_result, input_map

    @flow
    def run(self):
        return self.analyze_map.map(self.maps).result()

    @flow
    def analyze_new_maps(
        self, map_file: Path | str
    ) -> tuple[list, object, ProcessableMap]:
        from sotrplib.config.config import Settings

        input_data = Settings.from_file(map_file).to_basic()
        return self.analyze_map.map(input_data.maps).result()


@flow
def analyze_from_configuration(config: Path | str):
    from sotrplib.config.config import Settings

    pipeline = Settings.from_file(config).to_prefect()
    return pipeline.run()
