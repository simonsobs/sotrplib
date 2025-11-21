"""
Output data directly to lightserve.
"""

import httpx
from lightcurvedb.models.cutout import Cutout
from lightcurvedb.models.flux import FluxMeasurement
from soauth.toolkit.client import SOAuth
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sifter.core import SifterResult
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
        input_map: ProcessableMap,
    ):
        client = self.client()

        successful_uploads = 0
        total_uploads = 0

        for source in forced_photometry_candidates + sifter_result.source_candidates:
            response = client.put(
                "/api/v1/observations/",
                json={
                    "flux_measurement": FluxMeasurement(
                        band_name=source.frequency,
                        source_id=source.crossmatches[0].source_id,
                        time=source.observation_mean_time,
                        ra=source.ra.to_value("deg"),
                        dec=source.dec.to_value("deg"),
                        ra_uncertainty=source.err_ra.to_value("deg"),
                        dec_uncertainty=source.err_dec.to_value("deg"),
                        i_flux=source.flux.to_value("mJy"),
                        i_uncertainty=source.flux_uncertainty.to_value("mJy"),
                        extra={
                            "map_id": input_map.map_id,
                        },
                    ).model_dump_json(),
                    "cutout": Cutout(
                        data=source.cutout_data.data.tolist(),
                        time=source.observation_mean_time,
                        units=input_map.flux_units.to_string(),
                        band_name=source.frequency,
                        flux_id=None,
                    ).model_dump_json(),
                },
            )

            if not response.is_success:
                self.log.error(
                    "lightserve.output.failed",
                    status_code=response.status_code,
                    response_text=response.text,
                )
            else:
                successful_uploads += 1

            total_uploads += 1

        self.log.info(
            "lightserve.output.completed",
            successful_uploads=successful_uploads,
            total_uploads=total_uploads,
        )
