from typing import Literal, Optional

import numpy as np
import structlog
from astropy import units as u
from astropydantic import AstroPydanticQuantity, AstroPydanticTime, AstroPydanticUnit
from numpydantic import NDArray
from pixell import reproject
from pydantic import BaseModel, PrivateAttr
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap


class BaseSource(BaseModel):
    ra: AstroPydanticQuantity[u.deg]
    dec: AstroPydanticQuantity[u.deg]
    flux: Optional[AstroPydanticQuantity[u.mJy]] = None
    frequency: AstroPydanticQuantity[u.GHz] | None = None


class CrossMatch(BaseModel):
    ra: AstroPydanticQuantity[u.deg] | None = None
    dec: AstroPydanticQuantity[u.deg] | None = None
    observation_time: AstroPydanticTime | None = None
    source_id: str
    source_type: str | None = None
    probability: float | None = None
    angular_separation: AstroPydanticQuantity[u.deg] | None = None
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
        Literal["extragalactic", "star", "sso", "simulated", "unknown"] | None
    ) = None

    crossmatches: list[CrossMatch] | None = None
    catalog_name: str | None = None
    alternate_names: list[str] | None = None
    extended: bool | None = None
    err_ra: AstroPydanticQuantity[u.deg] | None = None
    err_dec: AstroPydanticQuantity[u.deg] | None = None
    err_flux: AstroPydanticQuantity[u.mJy] | None = None

    observation_start_time: AstroPydanticTime | None = None
    observation_mean_time: AstroPydanticTime | None = None
    observation_end_time: AstroPydanticTime | None = None

    flags: list[str] | None = None

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

    Simulated sources may also be stored in this class since we may wish to
    recover, or "measure", the injected parameters.

    """

    snr: float | None = None
    offset_ra: AstroPydanticQuantity[u.deg] | None = None
    offset_dec: AstroPydanticQuantity[u.deg] | None = None
    fwhm_ra: AstroPydanticQuantity[u.deg] | None = None
    fwhm_dec: AstroPydanticQuantity[u.deg] | None = None
    err_fwhm_ra: AstroPydanticQuantity[u.deg] | None = None
    err_fwhm_dec: AstroPydanticQuantity[u.deg] | None = None

    measurement_type: Literal["forced", "blind", "simulated"] = "forced"
    frequency: AstroPydanticQuantity[u.GHz] | None = None
    instrument: str | None = None
    array: str | None = None

    fit_method: Literal[
        "scipy_2d_gaussian",
        "lmfit_2d_gaussian",
        "nearest_neighbor",
        "spline",
        "simulated",
    ] = "lmfit_2d_gaussian"
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
        thumb_width: u.Quantity = 0.1 * u.deg,
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

        self.thumbnail_res = (
            thumb.wcs.wcs.cdelt * u.deg
        )  ## list of two values; 0 is RA, 1 is Dec
        self.thumbnail_unit = input_map.flux_units
        self.thumbnail = np.asarray(thumb)

    def __repr__(self):
        # Use vars(self) to get all attributes except thumbnail
        attrs = {k: v for k, v in vars(self).items() if k != "thumbnail"}
        return f"{self.__class__.__name__}({attrs})"
