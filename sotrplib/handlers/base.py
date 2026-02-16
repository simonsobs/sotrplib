from datetime import datetime, timezone
from typing import Iterable

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.database import set_processing_end
from sotrplib.maps.map_coadding import EmptyMapCoadder, MapCoadder
from sotrplib.maps.pointing import (
    EmptyPointingOffset,
    MapPointingOffset,
    save_model_maps,
)
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
    map_coadder: MapCoadder | None
    source_simulators: list[SimulatedSourceGenerator] | None
    source_injector: SourceInjector | None
    source_catalogs: list[SourceCatalog] | None
    sso_catalogs: list[SourceCatalog] | None
    preprocessors: list[MapPreprocessor] | None
    pointing_provider: ForcedPhotometryProvider | None
    pointing_residual_model: MapPointingOffset | None
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
        map_coadder: MapCoadder | None,
        source_simulators: list[SimulatedSourceGenerator] | None,
        source_injector: SourceInjector | None,
        source_catalogs: list[SourceCatalog] | None,
        sso_catalogs: list[SourceCatalog] | None,
        preprocessors: list[MapPreprocessor] | None,
        pointing_provider: ForcedPhotometryProvider | None,
        pointing_residual_model: MapPointingOffset | None,
        postprocessors: list[MapPostprocessor] | None,
        forced_photometry: ForcedPhotometryProvider | None,
        source_subtractor: SourceSubtractor | None,
        blind_search: BlindSearchProvider | None,
        sifter: SiftingProvider | None,
        outputs: list[SourceOutput] | None,
        profile: bool = False,
    ):
        self.maps = maps
        self.map_coadder = map_coadder or EmptyMapCoadder()
        self.source_simulators = source_simulators or []
        self.source_injector = source_injector or EmptySourceInjector()
        self.source_catalogs = source_catalogs or []
        self.sso_catalogs = sso_catalogs or []
        self.preprocessors = preprocessors or []
        self.pointing_provider = pointing_provider or EmptyForcedPhotometry()
        self.pointing_residual_model = pointing_residual_model or EmptyPointingOffset()
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
        if not np.any(output_map.hits > 0):
            return None
        for preprocessor in self.preprocessors:
            output_map = self.profilable_task(preprocessor.preprocess)(
                input_map=output_map
            )

        return output_map

    def coadd_maps(self, input_maps: list[ProcessableMap]) -> list[ProcessableMap]:
        return self.map_coadder.coadd(input_maps)

    @property
    def bbox(self):
        if not self.maps:
            return None

        # mapcat map list is not subscriptable so start with maximal bounding box
        bbox = [
            SkyCoord(ra=359.999 * units.deg, dec=90.0 * units.deg),
            SkyCoord(ra=0.0 * units.deg, dec=-90.0 * units.deg),
        ]
        for input_map in self.maps:
            map_bbox = input_map.bbox
            left = min(bbox[0].ra, map_bbox[0].ra)
            bottom = min(bbox[0].dec, map_bbox[0].dec)
            right = max(bbox[1].ra, map_bbox[1].ra)
            top = max(bbox[1].dec, map_bbox[1].dec)
            bbox = [SkyCoord(ra=left, dec=bottom), SkyCoord(ra=right, dec=top)]
        return bbox

    @property
    def observation_time_range(self):
        if not self.maps:
            return (None, None)

        start_time = datetime.max.replace(tzinfo=timezone.utc)
        end_time = datetime.min.replace(tzinfo=timezone.utc)
        for input_map in self.maps:
            start_time = min(input_map.observation_start, start_time)
            end_time = max(input_map.observation_end, end_time)

        return (start_time, end_time)

    def simulate_sources(self) -> list[SimulatedSource]:
        """Generate sources based upon maximal bounding box of all maps"""
        if len(self.source_simulators) == 0:
            return []
        all_simulated_sources = []
        bbox = self.bbox
        time_range = self.observation_time_range
        for simulator in self.source_simulators:
            simulated_sources, catalog = self.profilable_task(simulator.generate)(
                box=bbox,
                time_range=time_range,
            )

            all_simulated_sources.extend(simulated_sources)
            self.source_catalogs.append(catalog)
        return all_simulated_sources

    def analyze_map(
        self, input_map: ProcessableMap, simulated_sources: list[SimulatedSource]
    ) -> tuple[list, object, ProcessableMap]:
        self.profilable_task(input_map.finalize)()

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

        pointing_data = self.profilable_task(self.pointing_residual_model.get_offset)(
            pointing_sources=pointing_sources
        )
        self.profilable_task(save_model_maps)(
            self.pointing_residual_model,
            pointing_data,
            input_map,
            filename_prefix=f"pointing_residual_{input_map.map_id}",
        )
        sso_sources = []
        for sso_catalog in self.sso_catalogs:
            sso_sources.extend(
                self.profilable_task(sso_catalog.get_sources_in_map)(
                    input_map=input_map
                )
            )

        forced_photometry_candidates = self.profilable_task(
            self.forced_photometry.force
        )(
            input_map=input_map,
            catalogs=self.source_catalogs + self.sso_catalogs,
            pointing_residuals=self.pointing_residual_model,
            pointing_offset_data=pointing_data,
        )

        source_subtracted_map = self.profilable_task(self.source_subtractor.subtract)(
            sources=forced_photometry_candidates, input_map=input_map
        )

        blind_sources, _ = self.profilable_task(self.blind_search.search)(
            input_map=source_subtracted_map,
            pointing_residuals=self.pointing_residual_model,
            pointing_offset_data=pointing_data,
        )

        sifter_result = self.profilable_task(self.sifter.sift)(
            sources=blind_sources,
            catalogs=self.source_catalogs + self.sso_catalogs,
            input_map=source_subtracted_map,
        )

        for output in self.outputs:
            self.profilable_task(output.output)(
                forced_photometry_candidates=forced_photometry_candidates,
                sifter_result=sifter_result,
                input_map=input_map,
                pointing_sources=pointing_sources,
                injected_sources=injected_sources,
            )
        if input_map._parent_database is not None:
            set_processing_end(input_map.map_id)
        return forced_photometry_candidates, sifter_result, input_map

    def run(self) -> tuple[list[list], list[object], list[ProcessableMap]]:
        return self.flow(self._run)()

    def _run(self) -> tuple[list[list], list[object], list[ProcessableMap]]:
        """
        The actual pipeline run logic has to be in a separate method so that it can be
        decorated with the flow as prefect needs these to be defined in advance.
        """
        all_simulated_sources = self.basic_task(self.simulate_sources)()
        self.maps = self.basic_task(self.build_map).map(self.maps).result()
        self.maps = [m for m in self.maps if m is not None]
        self.maps = self.coadd_maps(self.maps)
        return (
            self.basic_task(self.analyze_map)
            .map(self.maps, self.unmapped(all_simulated_sources))
            .result()
        )
