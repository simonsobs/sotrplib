"""
A basic pipeline handler. Takes all the components and runs
them in the pre-specified order.
"""

from astropy import units as u
from astropy.coordinates import SkyCoord

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.postprocessor import MapPostprocessor
from sotrplib.maps.preprocessor import MapPreprocessor
from sotrplib.outputs.core import SourceOutput
from sotrplib.sifter.core import EmptySifter, SiftingProvider
from sotrplib.sims.sim_source_generators import SimulatedSourceGenerator
from sotrplib.sims.source_injector import EmptySourceInjector, SourceInjector
from sotrplib.source_catalog.database import MockDatabase
from sotrplib.sources.blind import EmptyBlindSearch
from sotrplib.sources.core import BlindSearchProvider, ForcedPhotometryProvider
from sotrplib.sources.force import EmptyForcedPhotometry
from sotrplib.sources.subtractor import EmptySourceSubtractor, SourceSubtractor


class PipelineRunner:
    def __init__(
        self,
        maps: list[ProcessableMap],
        source_simulators: list[SimulatedSourceGenerator] | None,
        source_injector: SourceInjector | None,
        source_catalogs: list[MockDatabase] | None,
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

    def run(self):
        for input_map in self.maps:
            input_map.build()

            for preprocessor in self.preprocessors:
                preprocessor.preprocess(input_map=input_map)

        # Generate sources based upon maximal bounding box of all maps
        bbox = [
            SkyCoord(ra=0.0 * u.deg, dec=-90.0 * u.deg),
            SkyCoord(ra=360.0 * u.deg, dec=90.0 * u.deg),
        ]

        for input_map in self.maps:
            map_bbox = input_map.bbox
            bbox[0].ra = min(bbox[0].ra, map_bbox[0].ra)
            bbox[0].dec = min(bbox[0].dec, map_bbox[0].dec)
            bbox[1].ra = max(bbox[1].ra, map_bbox[1].ra)
            bbox[1].dec = max(bbox[1].dec, map_bbox[1].dec)

        all_simulated_sources = []

        for simulator in self.source_simulators:
            simulated_sources, catalog = simulator.generate_sources(bbox=bbox)

            all_simulated_sources.extend(simulated_sources)
            self.source_catalogs.extend(catalog)

        for input_map in self.maps:
            input_map = self.source_injector.inject(
                input_map=input_map, simulated_sources=all_simulated_sources
            )

            input_map.finalize()

            for postprocessor in self.postprocessors:
                postprocessor.postprocess(input_map=input_map)

            forced_photometry_candidates = self.forced_photometry.force(
                input_map=input_map, catalogs=self.source_catalogs
            )

            source_subtracted_map = self.source_subtractor.subtract(
                sources=forced_photometry_candidates, input_map=input_map
            )

            blind_sources, _ = self.blind_search.search(input_map=source_subtracted_map)

            sifter_result = self.sifter.sift(
                catalogs=self.source_catalogs, input_map=source_subtracted_map
            )

            for output in self.outputs:
                output.output(
                    forced_photometry_candidates=forced_photometry_candidates,
                    sifter_result=sifter_result,
                    input_map=input_map,
                )

        return
