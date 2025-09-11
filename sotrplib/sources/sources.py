from typing import Literal, Optional

import numpy as np
import structlog
from astropy import units as u
from astropydantic import AstroPydanticQuantity, AstroPydanticUnit
from numpydantic import NDArray
from pixell import reproject
from pydantic import BaseModel, PrivateAttr
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap


class BaseSource(BaseModel):
    ra: AstroPydanticQuantity[u.deg]
    dec: AstroPydanticQuantity[u.deg]
    flux: Optional[AstroPydanticQuantity[u.mJy]] = None


class CrossMatch(BaseModel):
    source_id: str
    probability: float | None = None
    distance: AstroPydanticQuantity[u.deg] | None = None
    flux: AstroPydanticQuantity[u.mJy] | None = None
    err_flux: AstroPydanticQuantity[u.mJy] | None = None
    frequency: AstroPydanticQuantity[u.GHz] | None = None
    catalog_name: str | None = None
    catalog_idx: int | None = None
    alternate_names: list[str] | None = None


class RegisteredSource(BaseSource):
    """
    A registered source object which may have catalog crossmatches.
    This object has a source_id which may belong to the catalog and a source type.

    This object need not have a flux.

    Extendedness and positional uncertainty is stored if available.
    """

    source_id: str | None = None
    source_type: (
        Literal["extragalactic", "star", "asteroid", "simulated", "unknown"] | None
    ) = None

    crossmatches: list[CrossMatch] | None = None

    extended: bool | None = None
    err_ra: AstroPydanticQuantity[u.deg] | None = None
    err_dec: AstroPydanticQuantity[u.deg] | None = None
    err_flux: AstroPydanticQuantity[u.mJy] | None = None
    _log: FilteringBoundLogger = PrivateAttr(default_factory=structlog.get_logger)

    def add_crossmatch(self, crossmatch: CrossMatch):
        """
        Add a crossmatch object.
        Simple wrapper to append the crossmatch to list of crossmatches.
        """
        if self.crossmatches is None:
            self.crossmatches = []
        self.crossmatches.append(crossmatch)
        self._log.info("registeredsource.added_crossmatch", crossmatch=crossmatch)

    def update_crossmatches(
        self,
        crossmatches: list[CrossMatch],
        replace: bool = False,
    ):
        """
        Update the crossmatch info.

        Need to figure out what we want to do here...

        """
        raise NotImplementedError("Haven't figured out what updating means")


class MeasuredSource(RegisteredSource):
    """
    A source object specifically for a measurement in a given map.

    Since it's being measured, it is assumed to be a RegisteredSource which
    undergoes a particular forced photometry style flux measurement.

    """

    snr: float | None = None
    offset_ra: AstroPydanticQuantity[u.deg] | None = None
    offset_dec: AstroPydanticQuantity[u.deg] | None = None
    fwhm_ra: AstroPydanticQuantity[u.deg] | None = None
    fwhm_dec: AstroPydanticQuantity[u.deg] | None = None
    err_fwhm_ra: AstroPydanticQuantity[u.deg] | None = None
    err_fwhm_dec: AstroPydanticQuantity[u.deg] | None = None

    measurement_type: Literal["forced", "blind"] = "forced"
    frequency: AstroPydanticQuantity[u.GHz] | None = None

    fit_method: Literal["2d_gaussian", "nearest_neighbor", "spline"] = "2d_gaussian"
    fit_params: dict | None = None
    fit_failed: bool = False
    fit_failure_reason: str | None = None

    thumbnail: NDArray | None = None
    thumbnail_res: AstroPydanticQuantity[u.arcmin] | None = None
    thumbnail_unit: AstroPydanticUnit | None = None

    _log: FilteringBoundLogger = PrivateAttr(default_factory=structlog.get_logger)

    def extract_thumbnail(
        self,
        input_map: ProcessableMap,
        thumb_width: u.Quantity = 0.25 * u.deg,
        reproject_thumb=False,
    ):
        """
        Extract a thumbnail from the source's map.
        """
        if reproject_thumb:
            thumb = reproject.thumbnails(
                input_map.flux,
                [self.dec.to(u.rad).value, self.ra.to(u.rad).value],
                r=thumb_width.to(u.rad).value,
                res=input_map.map_resolution.to(u.rad).value,
            )
        else:
            thumb = input_map.flux.submap(
                [
                    [
                        self.dec.to(u.rad).value - thumb_width.to(u.rad).value,
                        self.ra.to(u.rad).value - thumb_width.to(u.rad).value,
                    ],
                    [
                        self.dec.to(u.rad).value + thumb_width.to(u.rad).value,
                        self.ra.to(u.rad).value + thumb_width.to(u.rad).value,
                    ],
                ],
            )
        self.thumbnail_res = input_map.map_resolution
        self.thumbnail_unit = input_map.flux_units
        self.thumbnail = np.asarray(thumb)
