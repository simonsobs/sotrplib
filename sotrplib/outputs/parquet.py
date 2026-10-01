"""
Write direct outputs to parquet files so that they can be easily
ingested into lightcurvedb by an external process.
"""

import datetime
import itertools
from pathlib import Path

import pandas as pd
from astropy.time import TimezoneInfo
from structlog import get_logger
from structlog.types import FilteringBoundLogger
from uuid7 import UUID as UUID7

from sotrplib.sifter.core import SifterResult
from sotrplib.sims.sim_sources import SimulatedSource
from sotrplib.sources.sources import MeasuredSource

from .core import SourceOutput

UTC = TimezoneInfo(tzname="utc")


class ParquetOutput(SourceOutput):
    def __init__(
        self,
        directory: Path,
        output_source_candidates: bool = True,
        output_transient_candidates: bool = True,
        output_noise_candidates: bool = False,
        log: FilteringBoundLogger | None = None,
    ):
        self.directory = directory
        self.output_source_candidates = output_source_candidates
        self.output_transient_candidates = output_transient_candidates
        self.output_noise_candidates = output_noise_candidates
        self.log = log or get_logger()

    def _get_map_stub_for_filename(self, map_name: str, mapcat_id: UUID7 | None) -> str:
        """
        Get a stub for the filename based on the map name and mapcat_id.
        Prefer the human-readable map_name for filenames; fall back to the
        real mapcat_id (stringified) if a name isn't available.
        """
        if mapcat_id is not None:
            return str(mapcat_id)
        if map_name:
            return map_name

        return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    def _sources_filename(self, map_id: UUID7 | None, map_name: str) -> Path:
        name = self._get_map_stub_for_filename(map_name=map_name, mapcat_id=map_id)
        return self.directory / f"{name}_sources.parquet"

    def _lightcurve_filename(self, map_id: UUID7 | None, map_name: str) -> Path:
        name = self._get_map_stub_for_filename(map_name=map_name, mapcat_id=map_id)
        return self.directory / f"{name}_lightcurve.parquet"

    def _cutout_filename(self, map_id: UUID7 | None, map_name: str) -> Path:
        name = self._get_map_stub_for_filename(map_name=map_name, mapcat_id=map_id)
        return self.directory / f"{name}_cutouts.parquet"

    def _ancillary_sources_filename(self, map_id: UUID7 | None, map_name: str) -> Path:
        name = self._get_map_stub_for_filename(map_name=map_name, mapcat_id=map_id)
        return self.directory / f"{name}_ancillary_sources.parquet"

    def create_sources(self, measured: list[MeasuredSource]) -> pd.DataFrame:
        """
        Create a DataFrame from the measured sources.
        """

        output_data = {}

        for data in measured:
            for match in data.crossmatches:
                if match.source_id in output_data:
                    continue

                output_data[match.source_id] = {
                    "ra": match.ra.to_value("deg"),
                    "dec": match.dec.to_value("deg"),
                    "source_id": match.source_id,
                    "name": match.catalog_name,
                    "variable": True,
                    "extra": {
                        "source_type": data.source_type,
                        "alternate_names": data.alternate_names,
                    },
                }

        df = pd.DataFrame.from_dict(output_data.values(), orient="columns")

        return df

    def create_lightcurve(
        self, measured: list[MeasuredSource], map_id: UUID7 | None
    ) -> pd.DataFrame:
        """
        Create a DataFrame of flux data from the measured sources.
        """

        output_data = []

        for data in measured:
            if data.flux is None:
                continue

            output_data.append(
                {
                    "ra": data.ra.to_value("deg"),
                    "dec": data.dec.to_value("deg"),
                    "ra_uncertainty": data.err_ra.to_value("deg")
                    if data.err_ra is not None
                    else None,
                    "dec_uncertainty": data.err_dec.to_value("deg")
                    if data.err_dec is not None
                    else None,
                    "source_id": data.source_id,
                    "name": data.catalog_name,
                    "flux": data.flux.to_value("mJy"),
                    "flux_err": data.err_flux.to_value("mJy")
                    if data.err_flux is not None
                    else None,
                    "time": data.observation_mean_time.to_datetime(timezone=UTC)
                    if data.observation_mean_time is not None
                    else None,
                    "measurement_id": data.measurement_id,
                    "frequency": int(data.frequency.to_value("GHz"))
                    if data.frequency is not None
                    else None,
                    "module": data.array,
                    "map_id": map_id,
                }
            )

        df = pd.DataFrame.from_dict(output_data, orient="columns")

        return df

    def create_sifter_lightcurve(
        self, sifter_result: SifterResult, map_id: UUID7 | None
    ) -> pd.DataFrame:
        """
        Create a DataFrame of flux data from the sifter data.
        """

        output_data = []

        to_chain = []

        if self.output_source_candidates:
            to_chain.append(sifter_result.source_candidates)

        if self.output_transient_candidates:
            to_chain.append(sifter_result.transient_candidates)

        if self.output_noise_candidates:
            to_chain.append(sifter_result.noise_candidates)

        for data in itertools.chain(*to_chain):
            if data.flux is None:
                continue

            output_data.append(
                {
                    "ra": data.ra.to_value("deg"),
                    "dec": data.dec.to_value("deg"),
                    "ra_uncertainty": data.err_ra.to_value("deg")
                    if data.err_ra is not None
                    else None,
                    "dec_uncertainty": data.err_dec.to_value("deg")
                    if data.err_dec is not None
                    else None,
                    "source_id": data.source_id,
                    "name": data.catalog_name,
                    "flux": data.flux.to_value("mJy"),
                    "flux_err": data.err_flux.to_value("mJy")
                    if data.err_flux is not None
                    else None,
                    "time": data.observation_mean_time.to_datetime(timezone=UTC)
                    if data.observation_mean_time is not None
                    else None,
                    "measurement_id": data.measurement_id,
                    "frequency": int(data.frequency.to_value("GHz"))
                    if data.frequency is not None
                    else None,
                    "module": data.array,
                    "map_id": map_id,
                }
            )

        df = pd.DataFrame.from_dict(output_data, orient="columns")

        return df

    def create_cutouts(
        self, measured: list[MeasuredSource], map_id: UUID7 | None
    ) -> pd.DataFrame:
        """
        Create a DataFrame of cutout data from the measured sources.
        """

        output_data = []

        for data in measured:
            output_data.append(
                {
                    "source_id": data.source_id,
                    "measurement_id": data.measurement_id,
                    "name": data.catalog_name,
                    "time": data.observation_mean_time.to_datetime(timezone=UTC)
                    if data.observation_mean_time is not None
                    else None,
                    "data": data.thumbnail.tolist()
                    if data.thumbnail is not None
                    else None,
                    "units": str(data.thumbnail_unit),
                    "frequency": int(data.frequency.to_value("GHz"))
                    if data.frequency is not None
                    else None,
                    "module": data.array,
                    "map_id": map_id,
                }
            )

        df = pd.DataFrame.from_dict(output_data, orient="columns")

        return df

    def output(
        self,
        forced_photometry_candidates: list[MeasuredSource],
        sifter_result: SifterResult,
        map_name: str,
        mapcat_id: UUID7 | None = None,
        pointing_sources: list[MeasuredSource] = [],  # for compatibility
        injected_sources: list[SimulatedSource] = [],  # for compatibility
    ):
        """
        Output the source candidates somehow. We also pass the
        input map in case e.g. we wish to reconstruct thumbnails from it.
        """

        output_sources = self.create_sources(forced_photometry_candidates)
        output_sources.to_parquet(self._sources_filename(mapcat_id), index="source_id")

        output_lightcurve = self.create_lightcurve(
            forced_photometry_candidates, map_id=mapcat_id
        )
        output_sifter_lightcurve = self.create_sifter_lightcurve(
            sifter_result, map_id=mapcat_id
        )
        combined_df = pd.concat(
            [output_lightcurve, output_sifter_lightcurve], ignore_index=True
        )
        combined_df.to_parquet(
            self._lightcurve_filename(mapcat_id), index="measurement_id"
        )

        output_cutouts = self.create_cutouts(
            forced_photometry_candidates, map_id=mapcat_id
        )
        output_cutouts.to_parquet(
            self._cutout_filename(mapcat_id), index="measurement_id"
        )

        return
