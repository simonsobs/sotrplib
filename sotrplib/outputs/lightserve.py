"""
Output data directly to lightserve.
"""

import httpx
from lightcurvedb.models.cutout import Cutout
from lightcurvedb.models.flux import FluxMeasurementCreate
from soauth.toolkit.client import SOAuth
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.sifter.core import SifterResult
from sotrplib.sims.sim_sources import SimulatedSource
from sotrplib.sources.sources import MeasuredSource

from .core import SourceOutput


class LightServeOutput(SourceOutput):
    """
    Output source candidates to lightserve. Note that, for now, this only
    handles forced photometry output.
    """

    def __init__(
        self,
        hostname: str,
        token_tag: str | None = None,
        identity_server: str | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.hostname = hostname
        self.token_tag = token_tag
        self.identity_server = identity_server
        self.log = log or get_logger()

    def client(self):
        if self.token_tag:
            soauth = SOAuth(
                token_tag=self.token_tag, identity_server=self.identity_server
            )
        else:
            soauth = None

        return httpx.Client(base_url=self.hostname, auth=soauth)

    def output(
        self,
        forced_photometry_candidates: list[MeasuredSource],
        sifter_result: SifterResult,
        map_id: str,
        pointing_sources: list[MeasuredSource] = [],  # for compatibility
        injected_sources: list[SimulatedSource] = [],  # for compatibility
    ):
        client = self.client()

        successful_uploads = 0
        total_uploads = 0

        source_translations = client.get("/sources/").json()
        socat_to_internal = {
            st["socat_id"]: st["source_id"] for st in source_translations
        }

        for source in forced_photometry_candidates + sifter_result.source_candidates:
            if not source.crossmatches:
                self.log.warning(
                    "lightserve.output.skipping_source_no_crossmatch",
                    ra=source.ra.to_value("deg"),
                    dec=source.dec.to_value("deg"),
                )
                continue

            fm = source.to_flux_measurement().model_dump_json()
            cut = source.to_cutout().model_dump_json()

            response = client.put(
                "/observations/",
                content="{" + f'"flux_measurement":{fm},"cutout":{cut}' + "}",
                headers={"Content-Type": "application/json"},
            )

            if not response.is_success:
                self.log.error(
                    "lightserve.output.failed",
                    status_code=response.status_code,
                    response_text=response.text,
                )
            else:
                self.log.info(
                    "lightserve.output.success",
                    crossmatches=source.crossmatches,
                    ra=source.ra.to_value("deg"),
                    dec=source.dec.to_value("deg"),
                )
                successful_uploads += 1

            total_uploads += 1

        self.log.info(
            "lightserve.output.completed",
            successful_uploads=successful_uploads,
            total_uploads=total_uploads,
        )
