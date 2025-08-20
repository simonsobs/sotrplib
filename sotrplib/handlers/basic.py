"""
A basic pipeline handler. Takes all the components and runs
them in the pre-specified order.
"""

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.postprocessor import MapPostprocessor
from sotrplib.maps.preprocessor import MapPreprocessor
from sotrplib.outputs.core import SourceOutput
from sotrplib.sifter.core import EmptySifter, SiftingProvider
from sotrplib.sources.core import (
    BlindSearchProvider,
    EmptyBlindSearch,
    EmptyForcedPhotometry,
    ForcedPhotometryProvider,
)
from sotrplib.sources.subtractor import EmptySourceSubtractor, SourceSubtractor


class PipelineRunner:
    def __init__(
        self,
        maps: list[ProcessableMap],
        preprocessors: list[MapPreprocessor] | None,
        postprocessors: list[MapPostprocessor] | None,
        forced_photometry: ForcedPhotometryProvider | None,
        source_subtractor: SourceSubtractor | None,
        blind_search: BlindSearchProvider | None,
        sifter: SiftingProvider | None,
        outputs: list[SourceOutput] | None,
    ):
        self.maps = maps
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

            input_map.finalize()

            for postprocessor in self.postprocessors:
                postprocessor.postprocess(input_map=input_map)

            forced_photometry_candidates, _ = self.forced_photometry.force(
                input_map=input_map
            )

            source_subtracted_map = self.source_subtractor.subtract(
                sources=forced_photometry_candidates, input_map=input_map
            )

            blind_sources, _ = self.blind_search.search(input_map=source_subtracted_map)

            # Should we be passing the source subtracted map or the original to the sifter..?
            sifter_result = self.sifter.sift(blind_sources, input_map)

            for output in self.outputs:
                output.output(
                    forced_photometry_candidates=forced_photometry_candidates,
                    sifter_result=sifter_result,
                    input_map=input_map,
                )

        return
