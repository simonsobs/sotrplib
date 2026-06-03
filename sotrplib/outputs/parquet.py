"""
Output straight to a parquet file that can be used
for asynchronous uploads to lightcurveDB or lightserve.
"""

import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.sifter.core import SifterResult
from sotrplib.sims.sim_sources import SimulatedSource
from sotrplib.sources.sources import MeasuredSource

from .core import SourceOutput


class ParquetOutput(SourceOutput):
    """
    Output source candidates to a parquet file. This can be used for
    asynchronous uploads to lightcurveDB or lightserve.
    """

    def __init__(
        self,
        directory: Path,
        log: FilteringBoundLogger | None = None,
    ):
        self.directory = directory
        self.log = log or get_logger()

        if not self.directory.exists():
            self.log.warning(
                "parquet_output.storage_path_does_not_exist",
                path=str(self.directory),
            )
            self.directory.mkdir(parents=True, exist_ok=True)

    def _filename(
        self, source_type: Literal["forced", "source", "transient", "noise"]
    ) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.directory / f"sotrplib_{timestamp}_{source_type}.parquet"

    def _serialize_to_file(
        self,
        sources: list[MeasuredSource],
        source_type: Literal["forced", "source", "transient", "noise"],
    ):
        """
        Builds the dataframe and writes out to file.
        """

        data = [x.to_flux_measurement().model_dump() for x in sources]

        if not data:
            self.log.info(
                "parquet_output.no_sources_to_write",
                source_type=source_type,
            )
            return

        df = pd.DataFrame(data)
        # Convert any UUID frames to strings for parquet compatibility
        for col in df.columns:
            if (
                df[col].dtype == "object"
                and df[col].apply(lambda x: hasattr(x, "hex")).any()
            ):
                df[col] = df[col].apply(
                    lambda x: x.hex if hasattr(x, "hex") else str(x)
                )

        df.set_index("measurement_id", inplace=True)
        filename = self._filename(source_type)
        df.to_parquet(filename)
        self.log.info(
            "parquet_output.wrote_file",
            path=str(filename),
            num_sources=len(sources),
            source_type=source_type,
        )

    def output(
        self,
        forced_photometry_candidates: list[MeasuredSource],
        sifter_result: SifterResult,
        map_id: str,
        pointing_sources: list[MeasuredSource] = [],  # for compatibility
        injected_sources: list[SimulatedSource] = [],  # for compatibility
    ):
        """
        Output to the filename specified in _filename. The format
        here is synchronized with what is required by the lightcurvedb.
        """

        self._serialize_to_file(forced_photometry_candidates, "forced")
        self._serialize_to_file(sifter_result.source_candidates, "source")
        self._serialize_to_file(sifter_result.transient_candidates, "transient")
        self._serialize_to_file(sifter_result.noise_candidates, "noise")

        return
