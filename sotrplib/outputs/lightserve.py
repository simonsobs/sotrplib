"""
Output data directly to lightserve.
"""

import httpx
from soauth.toolkit.client import SOAuth

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
    ):
        self.hostname = hostname
        self.token_tag = token_tag
        self.identity_server = identity_server

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

        raise NotImplementedError
