from typing import Any, Literal, Optional

import structlog
from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel, Field
from structlog.types import FilteringBoundLogger


class BaseSource(BaseModel):
    ra: AstroPydanticQuantity[u.deg]
    dec: AstroPydanticQuantity[u.deg]
    flux: Optional[AstroPydanticQuantity[u.mJy]] = None


class RegisteredSource(BaseSource):
    """
    A registered source object which may have catalog crossmatches.
    This object has a source_id which may belong to the catalog and a source type.

    This object need not have a flux.

    Extendedness and positional uncertainty is stored if available.
    """

    source_id: str | None = None
    source_type: Literal["Extragalactic", "Star", "Asteroid", "Unknown"] | None = None

    crossmatch_names: list | None = None
    crossmatch_probabilities: list | None = None
    crossmatch_fluxes: list | None = None
    crossmatch_frequencies: list | None = None

    extended: bool | None = None
    err_ra: AstroPydanticQuantity[u.deg] | None = None
    err_dec: AstroPydanticQuantity[u.deg] | None = None
    log: FilteringBoundLogger | None = Field(default_factory=structlog.get_logger)

    model_config = {"arbitrary_types_allowed": True}

    def add_crossmatches(
        self,
        new_names: list,
        new_probabilities: list = None,
        new_fluxes: list = None,
        new_frequencies: list = None,
    ):
        """
        Add the crossmatch names and probabilities, fluxes, frequencies.
        """

        def equilibrate_list(values: list | None, default: Any = None):
            """Ensure list is initialized and filtered by crossmatch names."""
            if not values:
                values = [default] * len(new_names)
            assert len(values) == len(new_names)
            return [
                v
                for i, v in enumerate(values)
                if new_names[i] not in self.crossmatch_names
            ]

        ## Add the names first, ignore duplicates
        if not self.crossmatch_names:
            self.crossmatch_names = []

        new_names = [name for name in new_names if name not in self.crossmatch_names]

        # Use the helper for each list
        new_probabilities = equilibrate_list(new_probabilities, None)
        new_fluxes = equilibrate_list(new_fluxes, None)
        new_frequencies = equilibrate_list(new_frequencies, None)

        self.log.info(
            "registeredsource.add_crossmatches",
            new_names=new_names,
            new_probabilities=new_probabilities,
            new_fluxes=new_fluxes,
            new_frequencies=new_frequencies,
            RegisteredSource=self,
        )

        self.crossmatch_names.extend(new_names)
        if not self.crossmatch_probabilities:
            self.crossmatch_probabilities = new_probabilities
        else:
            self.crossmatch_probabilities.extend(new_probabilities)
        if not self.crossmatch_fluxes:
            self.crossmatch_fluxes = new_fluxes
        else:
            self.crossmatch_fluxes.extend(new_fluxes)
        if not self.crossmatch_frequencies:
            self.crossmatch_frequencies = new_frequencies
        else:
            self.crossmatch_frequencies.extend(new_frequencies)
        self.log.info("registeredsource.crossmatches_added", RegisteredSource=self)

        return

    def update_crossmatches(
        self,
        new_names: list,
        new_probabilities: list | None = None,
        new_fluxes: list | None = None,
        new_frequencies: list | None = None,
        replace: bool = False,
    ):
        """
        Update the crossmatch info.

        if probabilities, fluxes or frequencies is not None,
        must be same length as crossmatch_names
        """

        if not self.crossmatch_names:
            self.add_crossmatches(
                new_names, new_probabilities, new_fluxes, new_frequencies
            )
            return

        if new_probabilities is None:
            new_probabilities = [None] * len(new_names)
        assert len(new_probabilities) == len(new_names)
        if new_fluxes is None:
            new_fluxes = [None] * len(new_names)
        assert len(new_fluxes) == len(new_names)
        if new_frequencies is None:
            new_frequencies = [None] * len(new_names)
        assert len(new_frequencies) == len(new_names)

        self.log.info(
            "registeredsource.update_crossmatches",
            prev_crossmatch_names=self.crossmatch_names,
            prev_crossmatch_probabilities=self.crossmatch_probabilities,
            prev_crossmatch_fluxes=self.crossmatch_fluxes,
            prev_crossmatch_frequencies=self.crossmatch_frequencies,
            new_crossmatch_names=new_names,
            new_crossmatch_probabilities=new_probabilities,
            new_crossmatch_fluxes=new_fluxes,
            new_crossmatch_frequencies=new_frequencies,
        )

        if replace:
            self.crossmatch_names = new_names
            self.crossmatch_probabilities = new_probabilities
            self.crossmatch_fluxes = new_fluxes
            self.crossmatch_frequencies = new_frequencies
        else:
            for i, name in enumerate(new_names):
                if name not in self.crossmatch_names:
                    self.crossmatch_names.append(name)
                    self.crossmatch_probabilities.append(new_probabilities[i])
                    self.crossmatch_fluxes.append(new_fluxes[i])
                    self.crossmatch_frequencies.append(new_frequencies[i])
                else:
                    self.crossmatch_probabilities[self.crossmatch_names.index(name)] = (
                        new_probabilities[i]
                    )
                    self.crossmatch_fluxes[self.crossmatch_names.index(name)] = (
                        new_fluxes[i]
                    )
                    self.crossmatch_frequencies[self.crossmatch_names.index(name)] = (
                        new_frequencies[i]
                    )
        self.log.info("registeredsource.crossmatch_updated", RegisteredSource=self)
        return


class ForcedPhotometrySource(RegisteredSource):
    """
    A source object specifically for forced photometry measurements.

    Since it's forced photometry, it is assumed to be a RegisteredSource which
    undergoes a particular forced photometry flux measurement.

    """

    def __init__(
        self,
        ra: AstroPydanticQuantity[u.deg],
        dec: AstroPydanticQuantity[u.deg],
        flux: AstroPydanticQuantity[u.mJy] | None = None,
        err_flux: AstroPydanticQuantity[u.mJy] | None = None,
        source_id: str | None = None,
        source_type: Literal["Extragalactic", "Star", "Asteroid", "Unknown", "Galaxy"]
        | None = None,
        crossmatch_names: list | None = None,
        crossmatch_probabilities: list | None = None,
        extended: bool | None = None,
        err_ra: AstroPydanticQuantity[u.deg] | None = None,
        err_dec: AstroPydanticQuantity[u.deg] | None = None,
        fwhm_ra: AstroPydanticQuantity[u.deg] | None = None,
        fwhm_dec: AstroPydanticQuantity[u.deg] | None = None,
        fit_method: Literal["2D_Gaussian", "pixell.at", "other"] = "2D_Gaussian",
        fit_params: dict | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        super().__init__(
            ra,
            dec,
            flux,
            source_id,
            source_type,
            crossmatch_names,
            crossmatch_probabilities,
            extended,
            err_ra,
            err_dec,
            log=log,
        )
        self.fwhm_ra = fwhm_ra
        self.fwhm_dec = fwhm_dec
        self.err_flux = err_flux
        self.fit_method = fit_method
        self.fit_params = fit_params
        self.log = log
        self.log.info("forcedphotometrysource.created", ForcedPhotometrySource=self)


class BlindSearchSource(BaseSource):
    """
    A class for sources detected in the blind search.

    These don't a priori have a catalog counterpart.
    """

    def __init__(
        self,
        ra: u.Quantity,
        dec: u.Quantity,
        flux: u.Quantity | None = None,
        err_flux: u.Quantity | None = None,
        err_ra: u.Quantity | None = None,
        err_dec: u.Quantity | None = None,
        extract_algorithm: Literal["photutils_segmentation"]
        | None = "photutils_segmentation",
        extract_params: dict | None = None,
        fit_params: dict | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        super().__init__(ra, dec, flux)
        self.err_flux = err_flux
        self.err_ra = err_ra
        self.err_dec = err_dec
        self.extract_algorithm = extract_algorithm
        self.extract_params = extract_params
        self.fit_params = fit_params
        self.log = log
        self.log.info("blindsearchsource.created", BlindSearchSource=self)


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
