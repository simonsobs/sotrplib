"""
Output data directly to lightserve.
"""

import asyncio
from datetime import timezone
from uuid import UUID

import uuid7
from astropy.time import Time
from lightcurvedb.config import Settings as LightcurveDBSettings
from lightcurvedb.models.cutout import Cutout
from lightcurvedb.models.exceptions import SourceNotFoundException
from lightcurvedb.models.flux import FluxMeasurement
from lightcurvedb.models.source import Source
from structlog import get_logger
from structlog.types import FilteringBoundLogger

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

    async def _lookup_or_create_source_id(
        self,
        backend,
        socat_id: UUID,
        name: str,
        ra: float,
        dec: float,
    ) -> UUID | None:
        """
        Translate a socat_id to lightcurvedb's own internal source_id via
        the backend's indexed get_by_socat_id lookup, creating a new
        lightcurvedb source (with a fresh, unrelated internal source_id)
        if one doesn't exist yet and upsert_sources is enabled.
        """
        try:
            existing = await backend.sources.get_by_socat_id(socat_id)
            return existing.source_id
        except SourceNotFoundException:
            if not self.upsert_sources:
                return None

            source_id = await backend.sources.create(
                source=Source(
                    socat_id=socat_id,
                    name=name,
                    ra=ra,
                    dec=dec,
                    variable=False,
                    extra=None,
                )
            )

            await self.log.ainfo(
                "lightcurvedb.output.upserted_source",
                socat_id=socat_id,
                source_id=str(source_id),
            )
            return source_id

    async def _extract_and_upsert_sources(
        self, forced_photometry_candidates: list[MeasuredSource]
    ) -> dict[UUID, UUID]:
        source_translations: dict[UUID, UUID] = {}

        async with self.settings.backend as backend:
            for source in forced_photometry_candidates:
                if not source.crossmatches:
                    self.log.warning(
                        "lightcurvedb.output.skipping_source_no_crossmatch",
                        ra=source.ra.to_value("deg"),
                        dec=source.dec.to_value("deg"),
                    )
                    continue

                socat_id = source.crossmatches[0].catalog_idx

                if socat_id in source_translations:
                    continue

                source_id = await self._lookup_or_create_source_id(
                    backend,
                    socat_id,
                    source.crossmatches[0].source_id,
                    source.ra.to_value("deg"),
                    source.dec.to_value("deg"),
                )

                if source_id is not None:
                    source_translations[socat_id] = source_id

        return source_translations

    def _convert_internal_to_lightcurvedb_source(
        self,
        input_measurement: MeasuredSource,
        socat_to_internal: dict[UUID, UUID],
        map_time: Time | None = None,
        map_id: str | None = None,
    ) -> tuple[FluxMeasurement, Cutout] | None:
        if not input_measurement.crossmatches:
            self.log.warning(
                "lightcurvedb.output.skipping_source_no_crossmatch",
                ra=input_measurement.ra.to_value("deg"),
                dec=input_measurement.dec.to_value("deg"),
            )
            return None

        source_id = socat_to_internal.get(input_measurement.crossmatches[0].catalog_idx)

        if source_id is None:
            self.log.warning(
                "lightcurvedb.output.skipping_source_not_registered",
                socat_id=input_measurement.crossmatches[0].catalog_idx,
                ra=input_measurement.ra.to_value("deg"),
                dec=input_measurement.dec.to_value("deg"),
            )
            return None

        fm = FluxMeasurement(
            measurement_id=uuid7.create(),
            frequency=90,
            module="i1",
            source_id=source_id,
            time=(
                input_measurement.observation_mean_time.to_datetime()
                if input_measurement.observation_mean_time is not None
                else (
                    map_time.to_datetime(timezone=timezone.utc)
                    if map_time is not None
                    else None
                )
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
                    else (
                        map_time.to_datetime(timezone=timezone.utc)
                        if map_time is not None
                        else None
                    )
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
        socat_to_internal: dict[UUID, UUID],
        map_time: Time | None = None,
        map_id: str | None = None,
    ) -> tuple[list[FluxMeasurement], list[Cutout]]:
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
        flux_measurements: list[FluxMeasurement],
        cutouts: list[Cutout],
    ) -> int:
        async with self.settings.backend as backend:
            # create_batch() doesn't return the created IDs (neither the
            # postgres nor the parquet backend implementation does, despite
            # _upload_sources previously assuming otherwise) -- use each
            # FluxMeasurement's own measurement_id (client-generated above)
            # to link its cutout instead of relying on a return value.
            await backend.fluxes.create_batch(flux_measurements)
            processed_cutouts = [
                Cutout(
                    measurement_id=fm.measurement_id,
                    **cutout.model_dump(exclude={"measurement_id"}),
                )
                for fm, cutout in zip(flux_measurements, cutouts)
                if cutout is not None
            ]
            await backend.cutouts.create_batch(processed_cutouts)

        return len(flux_measurements)

    async def _flux_upload_flow(
        self,
        forced_photometry_candidates: list[MeasuredSource],
        map_time: Time | None = None,
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
        map_id: str,
        pointing_sources: list[MeasuredSource] = [],  # for compatibility
        injected_sources: list[SimulatedSource] = [],  # for compatibility
    ):
        total_uploads = len(forced_photometry_candidates)

        successful_uploads = asyncio.run(
            self._flux_upload_flow(
                forced_photometry_candidates,
                map_id=map_id,
            )
        )

        self.log.info(
            "lightcurvedb.output.completed",
            successful_uploads=successful_uploads,
            total_uploads=total_uploads,
        )
