from typing import Literal, Optional

import structlog
from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel, PrivateAttr
from structlog.types import FilteringBoundLogger


class BaseSource(BaseModel):
    ra: AstroPydanticQuantity[u.deg]
    dec: AstroPydanticQuantity[u.deg]
    flux: Optional[AstroPydanticQuantity[u.mJy]] = None


class CrossMatch(BaseModel):
    name: str
    probability: float | None = None
    distance: AstroPydanticQuantity[u.deg] | None = None
    flux: AstroPydanticQuantity[u.mJy] | None = None
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
    source_type: Literal["Extragalactic", "Star", "Asteroid", "Unknown"] | None = None

    crossmatches: list[CrossMatch] | None = None

    extended: bool | None = None
    err_ra: AstroPydanticQuantity[u.deg] | None = None
    err_dec: AstroPydanticQuantity[u.deg] | None = None
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

    fwhm_ra: AstroPydanticQuantity[u.deg] | None = None
    fwhm_dec: AstroPydanticQuantity[u.deg] | None = None
    fit_method: Literal["2d_gaussian", "pixell.at", "other"] = "2d_gaussian"
    fit_params: dict | None = None


class BlindSearchSource(BaseSource):
    """
    A class for sources detected in the blind search.

    These don't a priori have a catalog counterpart.
    """

    err_flux: AstroPydanticQuantity[u.mJy] | None = None
    err_ra: AstroPydanticQuantity[u.deg] | None = None
    err_dec: AstroPydanticQuantity[u.deg] | None = None
    extract_algorithm: Literal["photutils_segmentation"] | None = (
        "photutils_segmentation"
    )
    extract_params: dict | None = None
    fit_params: dict | None = None
    _log: FilteringBoundLogger = PrivateAttr(default_factory=structlog.get_logger)


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
