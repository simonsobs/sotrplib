from typing import Literal, Optional

import numpy as np
import structlog
import uuid7 as uuid
from astropy import units as u
from astropydantic import AstroPydanticQuantity, AstroPydanticTime, AstroPydanticUnit
from numpydantic import NDArray
from pixell import reproject
from pydantic import BaseModel, PrivateAttr
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap


class BaseSource(BaseModel):
    """Minimal source with sky position and optional flux."""

    ra: AstroPydanticQuantity[u.deg]
    dec: AstroPydanticQuantity[u.deg]
    flux: Optional[AstroPydanticQuantity[u.mJy]] = None
    frequency: AstroPydanticQuantity[u.GHz] | None = None


class CrossMatch(BaseModel):
    """A single catalog crossmatch result for a source."""

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
    catalog_idx: int | str | uuid.UUID | None = None
    alternate_names: list[str] | None = None


class RegisteredSource(BaseSource):
    """A catalog-registered source with optional crossmatches and metadata.

    Fields
    ------
    source_id : str, optional
        Catalog identifier for this source.
    source_type : str, optional
        Classification (``"extragalactic"``, ``"star"``, ``"sso"``,
        ``"simulated"``, or ``"unknown"``).
    crossmatches : list of CrossMatch, optional
        Catalog crossmatch results.
    catalog_name : str, optional
        Name of the originating catalog.
    alternate_names : list of str, optional
        Additional names for this source.
    extended : bool, optional
        Whether the source is spatially extended.
    err_ra : Quantity[deg], optional
        Positional uncertainty in RA.
    err_dec : Quantity[deg], optional
        Positional uncertainty in Dec.
    err_flux : Quantity[mJy], optional
        Flux uncertainty.
    observation_start_time : AstroPydanticTime, optional
        Start of the observing window.
    observation_mean_time : AstroPydanticTime, optional
        Mean observation time.
    observation_end_time : AstroPydanticTime, optional
        End of the observing window.
    flags : list of str, optional
        Quality or processing flags.
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
        """Append a crossmatch result to this source's crossmatch list.

        Parameters
        ----------
        crossmatch : CrossMatch
            Crossmatch result to add.
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
        """Update the crossmatch list for this source.

        Parameters
        ----------
        crossmatches : list of CrossMatch
            New crossmatch results.
        replace : bool, optional
            If ``True``, replace the existing list; otherwise merge.
        """
        raise NotImplementedError("Haven't figured out what updating means")


class MeasuredSource(RegisteredSource):
    """A source measurement produced by forced or blind photometry.

    Fields
    ------
    snr : float, optional
        Signal-to-noise ratio of the measurement.
    offset_ra : Quantity[deg], optional
        Fitted RA offset from the catalog position.
    offset_dec : Quantity[deg], optional
        Fitted Dec offset from the catalog position.
    fwhm_ra : Quantity[deg], optional
        Fitted FWHM in RA.
    fwhm_dec : Quantity[deg], optional
        Fitted FWHM in Dec.
    err_fwhm_ra : Quantity[deg], optional
        Uncertainty on ``fwhm_ra``.
    err_fwhm_dec : Quantity[deg], optional
        Uncertainty on ``fwhm_dec``.
    measurement_type : {"forced", "blind", "simulated"}
        How the flux was measured.
    fit_method : str
        Name of the fitting algorithm used.
    fit_params : dict, optional
        Full set of fit parameter values.
    fit_failed : bool
        Whether the fit failed.
    fit_failure_reason : str, optional
        Human-readable failure explanation.
    thumbnail : NDArray, optional
        Pixel values of the cutout used for fitting.
    thumbnail_res : Quantity[arcmin], optional
        Pixel scale of the thumbnail.
    thumbnail_unit : AstroPydanticUnit, optional
        Physical units of the thumbnail pixel values.
    """

    snr: float | None = None
    offset_ra: AstroPydanticQuantity[u.deg] | None = None
    offset_dec: AstroPydanticQuantity[u.deg] | None = None
    fwhm_ra: AstroPydanticQuantity[u.deg] | None = None
    fwhm_dec: AstroPydanticQuantity[u.deg] | None = None
    err_fwhm_ra: AstroPydanticQuantity[u.deg] | None = None
    err_fwhm_dec: AstroPydanticQuantity[u.deg] | None = None

    measurement_type: Literal["forced", "blind", "simulated"] = "forced"
    measurement_id: str | None = None
    frequency: AstroPydanticQuantity[u.GHz] | None = None
    instrument: str | None = None
    array: str | None = None

    fit_method: Literal[
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
        """Extract and store a flux thumbnail centred on this source.

        Parameters
        ----------
        input_map : ProcessableMap
            Map from which to cut the thumbnail.
        thumb_width : Quantity, optional
            Half-width of the thumbnail (default 0.1 deg).
        reproject_thumb : bool, optional
            If ``True``, reproject to a local tangent plane before cutting.
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
