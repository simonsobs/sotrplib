"""
Output data directly to lightserve.
"""

import asyncio
import datetime
from uuid import UUID

from lightcurvedb.config import Settings as LightcurveDBSettings
from lightcurvedb.models.cutout import Cutout
from lightcurvedb.models.flux import FluxMeasurementCreate
from lightcurvedb.models.source import Source
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sifter.core import SifterResult
from sotrplib.sims.sim_sources import SimulatedSource
from sotrplib.sources.sources import MeasuredSource

from .core import SourceOutput


class LightcurveDBOutput(SourceOutput):
    """
    Output source candidates to straight to lightcurveDB. Note that, for now,
    this only handles forced photometry output. This should be used for
    development or when you have direct access to the database server.
    """

    def __init__(
        self,
        settings: LightcurveDBSettings,
        upsert_sources: bool = False,
        log: FilteringBoundLogger | None = None,
    ):
        self.settings = settings
        self.upsert_sources = upsert_sources
        self.log = log or get_logger()

    async def _get_source_translations(self) -> dict[int, UUID]:
        async with self.settings.backend as backend:
            return {st.socat_id: st.source_id for st in await backend.sources.get_all()}

    async def _extract_and_upsert_sources(
        self, forced_photometry_candidates: list[MeasuredSource]
    ):
        source_translations = await self._get_source_translations()

        if not self.upsert_sources:
            return source_translations

        async with self.settings.backend as backend:
            for source in forced_photometry_candidates:
                if not source.crossmatches:
                    self.log.warning(
                        "lightcurvedb.output.skipping_source_no_crossmatch",
                        ra=source.ra.to_value("deg"),
                        dec=source.dec.to_value("deg"),
                    )
                    continue

                socat_id = int(source.crossmatches[0].catalog_idx)

                if socat_id not in source_translations:
                    source_id = await backend.sources.create(
                        source=Source(
                            socat_id=socat_id,
                            name=source.crossmatches[0].source_id,
                            ra=source.ra.to_value("deg"),
                            dec=source.dec.to_value("deg"),
                            variable=False,
                            extra=None,
                        )
                    )

                    source_translations[socat_id] = source_id

                    await self.log.ainfo(
                        "lightcurvedb.output.upserted_source",
                        socat_id=socat_id,
                        source_id=str(source_id),
                    )

        return source_translations

    def _convert_internal_to_lightcurvedb_source(
        self,
        input_measurement: MeasuredSource,
        socat_to_internal: dict[int, UUID],
        map_time: datetime.datetime,
        map_id: str | None = None,
    ) -> tuple[FluxMeasurementCreate, Cutout] | None:
        if not input_measurement.crossmatches:
            self.log.warning(
                "lightcurvedb.output.skipping_source_no_crossmatch",
                ra=input_measurement.ra.to_value("deg"),
                dec=input_measurement.dec.to_value("deg"),
            )
            return None

        source_id = socat_to_internal.get(
            int(input_measurement.crossmatches[0].catalog_idx)
        )

        fm = FluxMeasurementCreate(
            frequency=90,
            module="i1",
            source_id=source_id,
            time=(
                input_measurement.observation_mean_time.to_datetime()
                if input_measurement.observation_mean_time is not None
                else map_time
            ),
            ra=input_measurement.ra.to_value("deg"),
            dec=input_measurement.dec.to_value("deg"),
            ra_uncertainty=(
                input_measurement.err_ra.to_value("deg")
                if input_measurement.err_ra is not None
                else 0.0
            ),
            dec_uncertainty=(
                input_measurement.err_dec.to_value("deg")
                if input_measurement.err_dec is not None
                else 0.0
            ),
            flux=(
                input_measurement.flux.to_value("Jy")
                if input_measurement.flux is not None
                else 0.0
            ),
            flux_err=(
                input_measurement.err_flux.to_value("Jy")
                if input_measurement.err_flux is not None
                else 0.0
            ),
            extra={
                "map_id": (
                    input_measurement.map_id
                    if hasattr(input_measurement, "map_id")
                    else map_id
                ),
            },
        )

        cutout = (
            Cutout(
                data=input_measurement.thumbnail.tolist(),
                time=(
                    input_measurement.observation_mean_time.to_datetime()
                    if input_measurement.observation_mean_time is not None
                    else map_time
                ),
                units=(
                    input_measurement.flux.unit.to_string()
                    if input_measurement.flux is not None
                    else "Jy"
                ),
                frequency=90,
                module="i1",
                source_id=source_id,
            )
            if input_measurement.thumbnail is not None
            else None
        )

        return fm, cutout

    def _convert_all_sources(
        self,
        sources: list[MeasuredSource],
        socat_to_internal: dict[int, UUID],
        map_time: datetime.datetime,
        map_id: str | None = None,
    ) -> tuple[list[FluxMeasurementCreate], list[Cutout]]:
        flux_measurements = []
        cutouts = []

        for source in sources:
            result = self._convert_internal_to_lightcurvedb_source(
                source, socat_to_internal, map_time, map_id=map_id
            )
            if result is not None:
                fm, cutout = result
                flux_measurements.append(fm)
                cutouts.append(cutout)

        return flux_measurements, cutouts

    async def _upload_sources(
        self,
        flux_measurements: list[FluxMeasurementCreate],
        cutouts: list[Cutout],
    ) -> int:
        async with self.settings.backend as backend:
            flux_measurement_ids = await backend.fluxes.create_batch(flux_measurements)
            processed_cutouts = [
                Cutout(
                    measurement_id=fm_id,
                    **cutout.model_dump(exclude={"measurement_id"}),
                )
                for fm_id, cutout in zip(flux_measurement_ids, cutouts)
                if cutout is not None
            ]
            await backend.cutouts.create_batch(processed_cutouts)

        return len(flux_measurement_ids)

    async def _flux_upload_flow(
        self,
        forced_photometry_candidates: list[MeasuredSource],
        map_time: datetime.datetime,
        map_id: str | None = None,
    ):
        socat_to_internal = await self._extract_and_upsert_sources(
            forced_photometry_candidates=forced_photometry_candidates
        )
        flux_measurements, cutouts = self._convert_all_sources(
            sources=forced_photometry_candidates,
            socat_to_internal=socat_to_internal,
            map_time=map_time,
            map_id=map_id,
        )
        return await self._upload_sources(
            flux_measurements=flux_measurements, cutouts=cutouts
        )

    def output(
        self,
        forced_photometry_candidates: list[MeasuredSource],
        sifter_result: SifterResult,
        input_map: ProcessableMap,
        pointing_sources: list[MeasuredSource] = [],  # for compatibility
        injected_sources: list[SimulatedSource] = [],  # for compatibility
    ):
        total_uploads = len(forced_photometry_candidates)

        successful_uploads = asyncio.run(
            self._flux_upload_flow(
                forced_photometry_candidates,
                map_time=input_map.observation_time,
                map_id=input_map.map_id,
            )
        )

        self.log.info(
            "lightcurvedb.output.completed",
            successful_uploads=successful_uploads,
            total_uploads=total_uploads,
        )
