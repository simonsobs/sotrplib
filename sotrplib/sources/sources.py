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
from sotrplib.sources.finding import BlindSourceCandidate


class BaseSource(BaseModel):
    ra: AstroPydanticQuantity[u.deg]
    dec: AstroPydanticQuantity[u.deg]
    flux: Optional[AstroPydanticQuantity[u.mJy]] = None


class CrossMatch(BaseModel):
    name: str
    probability: float | None = None
    distance: AstroPydanticQuantity[u.deg] | None = None
    flux: AstroPydanticQuantity[u.mJy] | None = None
    err_flux: AstroPydanticQuantity[u.mJy] | None = None
    frequency: AstroPydanticQuantity[u.GHz] | None = None
    catalog_name: str | None = None
    catalog_idx: int | None = None


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


class ForcedPhotometrySource(RegisteredSource):
    """
    A source object specifically for forced photometry measurements.

    Since it's forced photometry, it is assumed to be a RegisteredSource which
    undergoes a particular forced photometry flux measurement.

    """

    err_flux: AstroPydanticQuantity[u.mJy] | None = None
    offset_ra: AstroPydanticQuantity[u.deg] | None = None
    offset_dec: AstroPydanticQuantity[u.deg] | None = None
    fwhm_ra: AstroPydanticQuantity[u.deg] | None = None
    fwhm_dec: AstroPydanticQuantity[u.deg] | None = None
    err_fwhm_ra: AstroPydanticQuantity[u.deg] | None = None
    err_fwhm_dec: AstroPydanticQuantity[u.deg] | None = None
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
        thumb_width: AstroPydanticQuantity[u.deg] = 0.25 * u.deg,
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

    @classmethod
    def from_blind_source_candidate(
        cls,
        candidate: BlindSourceCandidate,
    ) -> "ForcedPhotometrySource":
        """
        Create a ForcedPhotometrySource from a BlindSourceCandidate.
        """
        # Calculate the components of the FWHM along RA and Dec axes
        # orientation is in degrees, convert to radians
        theta = (
            candidate.orientation.to(u.rad).value
            if hasattr(candidate, "orientation") and candidate.orientation is not None
            else 0
        )
        a = candidate.semimajor_sigma
        b = candidate.semiminor_sigma
        # FWHM along RA: projection of ellipse along RA axis
        fwhm_ra = 2.355 * abs(a * np.cos(theta)) + abs(b * np.sin(theta))
        # FWHM along Dec: projection of ellipse along Dec axis
        fwhm_dec = 2.355 * abs(a * np.sin(theta)) + abs(b * np.cos(theta))

        return cls(
            ra=candidate.ra,
            dec=candidate.dec,
            flux=candidate.peakval,
            err_flux=candidate.peakval / candidate.peaksig,
            fwhm_ra=fwhm_ra,
            fwhm_dec=fwhm_dec,
        )


class SourceCandidate(BaseModel):
    ra: float
    dec: float
    err_ra: float
    err_dec: float
    flux: float
    err_flux: float
    snr: float
    freq: str
    ctime: float
    arr: str
    map_id: str = ""
    sourceID: str = ""
    source_type: str = ""
    matched_filtered: bool = False
    renormalized: bool = False
    catalog_crossmatch: bool = False
    crossmatch_names: list = []
    crossmatch_probabilities: list = []
    fit_type: str = "forced"
    fwhm_a: float = None
    fwhm_b: float = None
    err_fwhm_a: float = None
    err_fwhm_b: float = None
    orientation: float = None
    kron_flux: float = None
    kron_fluxerr: float = None
    kron_radius: float = None
    ellipticity: float = None
    elongation: float = None
    fwhm: float = None

    def update_crossmatches(
        self,
        match_names: list,
        match_probabilities: list = None,
    ):
        """
        Update the crossmatch names and probabilities with new matches.
        """
        self.crossmatch_names.extend(match_names)
        if match_probabilities is not None:
            self.crossmatch_probabilities.extend(match_probabilities)
        else:
            self.crossmatch_probabilities.extend([None] * len(match_names))

        return

    def to_forced_photometry_source(self) -> ForcedPhotometrySource:
        """
        Convert the SourceCandidate to a ForcedPhotometrySource.
        """
        return ForcedPhotometrySource(
            source_id=self.sourceID,
            flux=u.Quantity(self.flux, "Jy"),
            err_flux=u.Quantity(self.err_flux, "Jy"),
            ra=u.Quantity(self.ra, "deg"),
            dec=u.Quantity(self.dec, "deg"),
            err_ra=u.Quantity(self.err_ra, "deg"),
            err_dec=u.Quantity(self.err_dec, "deg"),
            fwhm_ra=u.Quantity(self.fwhm_a, "arcmin") if self.fwhm_a else None,
            fwhm_dec=u.Quantity(self.fwhm_b, "arcmin") if self.fwhm_b else None,
        )
