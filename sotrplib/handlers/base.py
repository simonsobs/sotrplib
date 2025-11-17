from typing import Iterable

from astropy.coordinates import SkyCoord

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.pointing import EmptyPointingOffset, MapPointingOffset
from sotrplib.maps.postprocessor import MapPostprocessor
from sotrplib.maps.preprocessor import MapPreprocessor
from sotrplib.outputs.core import SourceOutput
from sotrplib.sifter.core import EmptySifter, SiftingProvider
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
from sotrplib.sources.subtractor import EmptySourceSubtractor, SourceSubtractor

__all__ = ["BaseRunner"]


class BaseRunner:
    maps: Iterable[ProcessableMap]
    source_simulators: list[SimulatedSourceGenerator] | None
    source_injector: SourceInjector | None
    source_catalogs: list[SourceCatalog] | None
    preprocessors: list[MapPreprocessor] | None
    pointing_provider: ForcedPhotometryProvider | None
    pointing_residual: MapPointingOffset | None
    postprocessors: list[MapPostprocessor] | None
    forced_photometry: ForcedPhotometryProvider | None
    source_subtractor: SourceSubtractor | None
    blind_search: BlindSearchProvider | None
    sifter: SiftingProvider | None
    outputs: list[SourceOutput] | None
    profile: bool = False

    def __init__(
        self,
        maps: Iterable[ProcessableMap],
        source_simulators: list[SimulatedSourceGenerator] | None,
        source_injector: SourceInjector | None,
        source_catalogs: list[SourceCatalog] | None,
        preprocessors: list[MapPreprocessor] | None,
        pointing_provider: ForcedPhotometryProvider | None,
        pointing_residual: MapPointingOffset | None,
        postprocessors: list[MapPostprocessor] | None,
        forced_photometry: ForcedPhotometryProvider | None,
        source_subtractor: SourceSubtractor | None,
        blind_search: BlindSearchProvider | None,
        sifter: SiftingProvider | None,
        outputs: list[SourceOutput] | None,
        profile: bool = False,
    ):
        self.maps = maps
        self.source_simulators = source_simulators or []
        self.source_injector = source_injector or EmptySourceInjector()
        self.source_catalogs = source_catalogs or []
        self.preprocessors = preprocessors or []
        self.pointing_provider = pointing_provider or EmptyForcedPhotometry()
        self.pointing_residual = pointing_residual or EmptyPointingOffset()
        self.postprocessors = postprocessors or []
        self.forced_photometry = forced_photometry or EmptyForcedPhotometry()
        self.source_subtractor = source_subtractor or EmptySourceSubtractor()
        self.blind_search = blind_search or EmptyBlindSearch()
        self.sifter = sifter or EmptySifter()
        self.outputs = outputs or []
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

        for preprocessor in self.preprocessors:
            output_map = self.profilable_task(preprocessor.preprocess)(
                input_map=output_map
            )

        return output_map

    @property
    def bbox(self):
        if not self.maps:
            return None
        bbox = self.maps[0].bbox

        for input_map in self.maps[1:]:
            map_bbox = input_map.bbox
            left = min(bbox[0].ra, map_bbox[0].ra)
            bottom = min(bbox[0].dec, map_bbox[0].dec)
            right = max(bbox[1].ra, map_bbox[1].ra)
            top = max(bbox[1].dec, map_bbox[1].dec)
            bbox = [SkyCoord(ra=left, dec=bottom), SkyCoord(ra=right, dec=top)]
        return bbox

    def simulate_sources(self) -> list[SimulatedSource]:
        """Generate sources based upon maximal bounding box of all maps"""
        all_simulated_sources = []
        bbox = self.bbox
        for simulator in self.source_simulators:
            simulated_sources, catalog = self.profilable_task(simulator.generate)(
                box=bbox
            )

            all_simulated_sources.extend(simulated_sources)
            self.source_catalogs.append(catalog)
        return all_simulated_sources

    def analyze_map(
        self, input_map: ProcessableMap, simulated_sources: list[SimulatedSource]
    ) -> tuple[list, object, ProcessableMap]:
        self.profilable_task(input_map.finalize)()

        input_map = self.profilable_task(self.source_injector.inject)(
            input_map=input_map, simulated_sources=simulated_sources
        )

        for postprocessor in self.postprocessors:
            input_map = self.profilable_task(postprocessor.postprocess)(
                input_map=input_map
            )

        pointing_sources = self.profilable_task(self.pointing_provider.force)(
            input_map=input_map, catalogs=self.source_catalogs
        )

        _ = self.profilable_task(self.pointing_residual.get_offset)(
            pointing_sources=pointing_sources
        )

        forced_photometry_candidates = self.profilable_task(
            self.forced_photometry.force
        )(
            input_map=input_map,
            catalogs=self.source_catalogs,
            pointing_residuals=self.pointing_residual,
        )

        source_subtracted_map = self.profilable_task(self.source_subtractor.subtract)(
            sources=forced_photometry_candidates, input_map=input_map
        )

        blind_sources, _ = self.profilable_task(self.blind_search.search)(
            input_map=source_subtracted_map,
            pointing_residuals=self.pointing_residual,
        )

        sifter_result = self.profilable_task(self.sifter.sift)(
            sources=blind_sources,
            catalogs=self.source_catalogs,
            input_map=source_subtracted_map,
        )

        for output in self.outputs:
            self.profilable_task(output.output)(
                forced_photometry_candidates=forced_photometry_candidates,
                sifter_result=sifter_result,
                input_map=input_map,
            )

        return forced_photometry_candidates, sifter_result, input_map

    def run(self) -> tuple[list[list], list[object], list[ProcessableMap]]:
        return self.flow(self._run)()

    def _run(self) -> tuple[list[list], list[object], list[ProcessableMap]]:
        """
        The actual pipeline run logic has to be in a separate method so that it can be
        decorated with the flow as prefect needs these to be defined in advance.
        """
        self.maps = self.basic_task(self.build_map).map(self.maps).result()
        all_simulated_sources = self.basic_task(self.simulate_sources)()
        return (
            self.basic_task(self.analyze_map)
            .map(self.maps, self.unmapped(all_simulated_sources))
            .result()
        )
