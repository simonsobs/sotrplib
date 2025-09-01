from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pixell import enmap
from pydantic import BaseModel
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.core import BlindSearchProvider
from sotrplib.sources.finding import extract_sources
from sotrplib.sources.sources import BlindSearchSource


class EmptyBlindSearch(BlindSearchProvider):
    def search(
        self, input_map: ProcessableMap
    ) -> tuple[list[BlindSearchSource], list[enmap.ndmap]]:
        return [], []


class BlindSearchParameters(BaseModel):
    """
    Parameters for blind source searching using photutils.
    """

    sigma_threshold: float = 5.0
    minimum_separation: list[AstroPydanticQuantity[u.arcmin]] = [
        AstroPydanticQuantity(u.Quantity(0.5, "arcmin"))
    ]
    sigma_threshold_for_minimum_separation: list[float] = [3.0]


class SigmaClipBlindSearch(BlindSearchProvider):
    def __init__(
        self,
        log: FilteringBoundLogger | None = None,
        parameters: BlindSearchParameters | None = None,
        pixel_mask: enmap.ndmap | None = None,
    ):
        self.log = log or get_logger()
        self.parameters = parameters or BlindSearchParameters()
        self.pixel_mask = pixel_mask
        self.log.info("sigma_clip_blind_search.initialized", parameters=self.parameters)

    def search(
        self,
        input_map: ProcessableMap,
    ) -> tuple[list[BlindSearchSource], list[enmap.ndmap]]:
        if not input_map.finalized:
            raise ValueError(
                "Input map must be finalized before searching for sources."
            )
        if input_map.map_resolution is None:
            raise ValueError("Flux map must have a resolution.")

        self.log.info(
            "sigma_clip_blind_search.searching",
            nsigma=self.parameters.sigma_threshold,
            minrad=self.parameters.minimum_separation,
            sigma_thresh_for_minrad=self.parameters.sigma_threshold_for_minimum_separation,
        )
        extracted_sources = extract_sources(
            input_map,
            nsigma=self.parameters.sigma_threshold,
            minrad=self.parameters.minimum_separation,
            sigma_thresh_for_minrad=self.parameters.sigma_threshold_for_minimum_separation,
            log=self.log,
        )
        return extracted_sources, []
