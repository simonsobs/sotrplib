from typing import Any, Literal

from pydantic import BaseModel


class BaseSource(BaseModel):
    ra: float
    dec: float
    flux: float | None

    def __init__(self, ra, dec, flux=None):
        self.ra = ra
        self.dec = dec
        self.flux = flux

    def to_dict(self):
        return {
            "ra": self.ra,
            "dec": self.dec,
            "flux": self.flux,
        }


class RegisteredSource(BaseSource):
    """
    A registered source object which may have catalog crossmatches.
    This object has a sourceID which may belong to the catalog and a source type.

    This object need not have a flux.

    Extendedness and positional uncertainty is stored if available.
    """

    def __init__(
        self,
        ra: float,
        dec: float,
        flux: float | None = None,
        sourceID: str | None = None,
        source_type: Literal["Extragalactic", "Star", "Asteroid", "Unknown"]
        | None = None,
        crossmatch_names: list | None = None,
        crossmatch_probabilities: list | None = None,
        crossmatch_fluxes: list | None = None,
        crossmatch_frequencies: list | None = None,
        extended: bool | None = None,
        err_ra: float | None = None,
        err_dec: float | None = None,
    ):
        super().__init__(ra, dec, flux)
        self.sourceID = sourceID
        self.source_type = source_type
        self.crossmatch_names = crossmatch_names
        self.crossmatch_probabilities = crossmatch_probabilities
        self.crossmatch_fluxes = crossmatch_fluxes
        self.crossmatch_frequencies = crossmatch_frequencies
        self.extended = extended
        self.err_ra = err_ra
        self.err_dec = err_dec

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
            if values is None:
                values = [default] * len(new_names)
            assert len(values) == len(new_names)
            return [
                v
                for i, v in enumerate(values)
                if new_names[i] not in self.crossmatch_names
            ]

        # Use the helper for each list
        new_probabilities = equilibrate_list(new_probabilities, None)
        new_fluxes = equilibrate_list(new_fluxes, None)
        new_frequencies = equilibrate_list(new_frequencies, None)

        new_names = [name for name in new_names if name not in self.crossmatch_names]
        self.crossmatch_names.extend(new_names)
        self.crossmatch_probabilities.extend(new_probabilities)
        self.crossmatch_fluxes.extend(new_fluxes)
        self.crossmatch_frequencies.extend(new_frequencies)

        return

    def update_crossmatches(
        self,
        crossmatch_names: list,
        crossmatch_probabilities: list | None = None,
        crossmatch_fluxes: list | None = None,
        crossmatch_frequencies: list | None = None,
        replace: bool = False,
    ):
        """
        Update the crossmatch info.

        if probabilities, fluxes or frequencies is not None,
        must be same length as crossmatch_names
        """
        if crossmatch_probabilities is None:
            crossmatch_probabilities = [None] * len(crossmatch_names)
        assert len(crossmatch_probabilities) == len(crossmatch_names)
        if crossmatch_fluxes is None:
            crossmatch_fluxes = [None] * len(crossmatch_names)
        assert len(crossmatch_fluxes) == len(crossmatch_names)
        if crossmatch_frequencies is None:
            crossmatch_frequencies = [None] * len(crossmatch_names)
        assert len(crossmatch_frequencies) == len(crossmatch_names)

        if replace:
            self.crossmatch_names = crossmatch_names
            self.crossmatch_probabilities = crossmatch_probabilities
            self.crossmatch_fluxes = crossmatch_fluxes
            self.crossmatch_frequencies = crossmatch_frequencies
        else:
            for i, name in enumerate(crossmatch_names):
                if name not in self.crossmatch_names:
                    self.crossmatch_names.append(name)
                    self.crossmatch_probabilities.append(crossmatch_probabilities[i])
                    self.crossmatch_fluxes.append(crossmatch_fluxes[i])
                    self.crossmatch_frequencies.append(crossmatch_frequencies[i])
                else:
                    self.crossmatch_probabilities[self.crossmatch_names.index(name)] = (
                        crossmatch_probabilities[i]
                    )
                    self.crossmatch_fluxes[self.crossmatch_names.index(name)] = (
                        crossmatch_fluxes[i]
                    )
                    self.crossmatch_frequencies[self.crossmatch_names.index(name)] = (
                        crossmatch_frequencies[i]
                    )

        return


class ForcedPhotometrySource(RegisteredSource):
    """
    A source object specifically for forced photometry measurements.
    """

    def __init__(
        self,
        ra: float,
        dec: float,
        flux: float | None = None,
        sourceID: str | None = None,
        source_type: Literal["Extragalactic", "Star", "Asteroid", "Unknown", "Galaxy"]
        | None = None,
        crossmatch_names: list | None = None,
        crossmatch_probabilities: list | None = None,
        extended: bool | None = None,
        err_ra: float | None = None,
        err_dec: float | None = None,
    ):
        super().__init__(
            ra,
            dec,
            flux,
            sourceID,
            source_type,
            crossmatch_names,
            crossmatch_probabilities,
            extended,
            err_ra,
            err_dec,
        )

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict["fit_type"] = "forced"
        return base_dict


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
