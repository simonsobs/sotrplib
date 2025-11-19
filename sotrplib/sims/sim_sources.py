import math
from abc import ABC, abstractmethod
from datetime import timedelta

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropydantic import AstroPydanticQuantity
from pixell import enmap
from pydantic import AwareDatetime
from structlog.types import FilteringBoundLogger


class SimulatedSource(ABC):
    @abstractmethod
    def position(self, time: AwareDatetime) -> SkyCoord:
        return

    @abstractmethod
    def flux(self, time: AwareDatetime) -> u.Quantity:
        return


class FixedSimulatedSource(SimulatedSource):
    def __init__(
        self,
        position: SkyCoord,
        flux: u.Quantity,
    ):
        self._position = position
        self._flux = flux

        return

    def position(self, time):
        return self._position

    def flux(self, time):
        return self._flux


class GaussianTransientSimulatedSource(SimulatedSource):
    def __init__(
        self,
        position: SkyCoord,
        peak_time: AwareDatetime,
        flare_width: timedelta,
        peak_amplitude: u.Quantity = 0.0 * u.Jy,
    ):
        """
        Initialize a simulated source.

        Parameters:
        - position: Sky position of the source
        - peak_amplitude: The peak amplitude of the flare.
        - peak_time: The time at which the flare peaks.
        - flare_width: The FWHM of the flare (e.g., standard deviation for Gaussian).
        - flare_morph: The morphology of the flare ('Gaussian' supported for now).
        - beam_params: Dictionary of beam parameters (e.g., FWHM, ellipticity).
        """
        self._position = position
        self.peak_time = peak_time
        self.flare_width = flare_width
        self.peak_amplitude = peak_amplitude

        return

    def position(self, time: AwareDatetime) -> SkyCoord:
        return self._position

    def flux(self, time: AwareDatetime) -> u.Quantity:
        delta_time = time - self.peak_time

        sigma = self.flare_width / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        exponent = delta_time / sigma

        return self.peak_amplitude * math.exp(-0.5 * exponent * exponent)


class PowerLawTransientSimulatedSource(SimulatedSource):
    def __init__(
        self,
        position: SkyCoord,
        peak_time: datetime,
        peak_amplitude: u.Quantity,
        alpha_rise: float,
        alpha_decay: float,
        smoothness: float = 1.0,
        t_ref: timedelta = timedelta(days=1),
    ):
        """
        Initialize a power-law transient simulated source.

        Parameters:
            - position: Sky position of the source
            - peak_time: Time of maximum flux
            - peak_amplitude: Maximum flux of transient
            - alpha_rise: Early-time (rising) power-law index (>0)
            - alpha_decay: Late-time (decaying) power-law index (<0)
            - smoothness: Smoothness parameter (s). Controls how sharp (large s) or how rounded (small s) peak is.
            - t_ref: Reference timescale (default 1 day)
        """
        self._position = position
        self.peak_time = peak_time
        self.peak_amplitude = peak_amplitude
        self.alpha_rise = alpha_rise
        self.alpha_decay = alpha_decay
        self.smoothness = smoothness
        self.t_ref = t_ref

        # Calculate derived parameters
        # Find tau at peak: tau_peak = (alpha_rise / |alpha_decay|)^(1/(s*(alpha_rise + |alpha_decay|)))
        s = self.smoothness
        abs_slpha_decay = abs(alpha_decay)
        tau_peak = (alpha_rise / abs_slpha_decay) ** (
            1.0 / (s * (alpha_rise + abs_slpha_decay))
        )

        # Calculate onset time: t_0 = t_peak - tau_peak * t_ref
        self.onset_time = peak_time - timedelta(
            seconds=tau_peak * t_ref.total_seconds()
        )

        # Calculate S0 from peak amplitude
        # F_peak = S0 * [tau_peak^(s*alpha_rise) + tau_peak^(s*alpha_decay)]^(-1/s)
        term1 = tau_peak ** (s * alpha_rise)
        term2 = tau_peak ** (s * alpha_decay)
        self.S0 = peak_amplitude * (term1 + term2) ** (1.0 / s)

        return

    def position(self, time: datetime) -> SkyCoord:
        return self._position

    def flux(self, time: datetime) -> u.Quantity:
        # Return zero flux before onset time
        if time < self.onset_time:
            return 0.0 * self.peak_amplitude.unit

        # Calculate tau = (t - t0) / t_ref
        delta_time = time - self.onset_time
        tau = delta_time.total_seconds() / self.t_ref.total_seconds()

        # Compute the smoothly broken power law
        # S(t) = S0 * [tau^(s*alpha_rise) + tau^(s*alpha_decay)]^(-1/s)
        s = self.smoothness
        term1 = tau ** (s * self.alpha_rise)
        term2 = tau ** (s * self.alpha_decay)

        flux = self.S0 * (term1 + term2) ** (-1.0 / s)
        return flux


class SimTransient:
    pass


def generate_transients(
    n: int = None,
    imap: enmap.ndmap = None,
    ra_lims: AstroPydanticQuantity[u.deg] | None = None,
    dec_lims: AstroPydanticQuantity[u.deg] | None = None,
    positions: AstroPydanticQuantity[u.deg] | None = None,
    peak_amplitudes: AstroPydanticQuantity[u.Jy] | None = None,
    peak_times: list | None = None,
    flare_widths: list | None = None,
    flare_morphs: list = None,
    beam_params: list = None,
    uniform_on_sky=False,
    log: FilteringBoundLogger | None = None,
):
    """
    Generate a list of simulated transient sources.
    These will be either generated uniformly on sky or uniformly in the flatsky map.
    If imap is given, only inject sources within the weighted region.
    Each source generates a SimTransient object.


    Arguments:
    - n (int): Number of transients to generate. If None, positions must be provided.
    - imap (enmap.ndmap): Input map for generating random positions. If None, ra_lims and dec_lims must be provided.
    - ra_lims (tuple): Tuple of (min_ra, max_ra) for random RA generation. degrees
    - dec_lims (tuple): Tuple of (min_dec, max_dec) for random Dec generation. degrees
    - positions (list): List of tuples (ra, dec) for specific positions. If provided, n is ignored. degrees
    - peak_amplitudes (list|tuple): List of peak amplitudes for each transient or tuple of (min, max) for random generation.
    - peak_times (list|tuple): List of peak times for each transient or tuple of (min, max) for random generation.
    - flare_widths (list|tuple): List of flare widths for each transient or tuple of (min, max) for random generation.
    - flare_morphs (list): List of flare morphologies for each transient.
    - beam_params (list): List of dictionaries containing beam parameters for each transient.
    - uniform_on_sky (bool): generate random positions uniform on sky or uniform on imap flatsky
    """
    from .sim_utils import (
        generate_random_flare_amplitudes,
        generate_random_flare_times,
        generate_random_flare_widths,
        generate_random_positions,
        generate_random_positions_in_map,
    )

    transients = []
    if n is None and positions is None:
        raise ValueError("Either n or positions must be provided.")
    if n is not None and positions is not None:
        raise ValueError("Cannot provide both n and positions.")
    n_positions = None
    if n is not None:
        n_positions = 0
        positions = []

        ntries = 10
        while n_positions < n and ntries > 0:
            if uniform_on_sky or imap is None:
                rand_pos = generate_random_positions(
                    n, imap=imap, ra_lims=ra_lims, dec_lims=dec_lims
                )
            else:
                rand_pos = generate_random_positions_in_map(n, imap)

            if isinstance(imap, enmap.ndmap):
                for p in rand_pos:
                    if (
                        imap.at([p[0].to_value(u.rad), p[1].to_value(u.rad)], mode="nn")
                        and len(positions) < n
                    ):
                        positions.append(p)
            else:
                positions = rand_pos

            n_positions = len(positions)
            ntries -= 1
            if ntries == 0 and n_positions < n:
                print(
                    f"Failed to inject {n} sources into weighted. Only injected {n_positions}"
                )
    if n_positions is not None:
        n = n_positions
    if n is not None and isinstance(peak_amplitudes, tuple):
        peak_amplitudes = generate_random_flare_amplitudes(
            n, min_amplitude=peak_amplitudes[0], max_amplitude=peak_amplitudes[1]
        )
    elif n is not None and isinstance(peak_amplitudes, type(None)):
        peak_amplitudes = generate_random_flare_amplitudes(n)

    if n is not None and isinstance(peak_times, tuple):
        peak_times = generate_random_flare_times(
            n, start_time=peak_times[0], end_time=peak_times[1]
        )
    elif n is not None and isinstance(peak_times, type(None)):
        peak_times = generate_random_flare_times(n)

    if n is not None and isinstance(flare_widths, tuple):
        flare_widths = generate_random_flare_widths(
            n, min_width=flare_widths[0], max_width=flare_widths[1]
        )
    elif n is not None and isinstance(flare_widths, type(None)):
        flare_widths = generate_random_flare_widths(n)

    if n is not None and isinstance(flare_morphs, type(None)):
        flare_morphs = ["Gaussian"] * n

    if n is not None and isinstance(beam_params, type(None)):
        beam_params = [{}] * n

    if (
        len(positions) != len(peak_amplitudes)
        or len(positions) != len(peak_times)
        or len(positions) != len(flare_widths)
    ):
        raise ValueError("All input lists must be of the same length.")

    for i in range(len(positions)):
        if flare_morphs[i] == "Gaussian":
            transient = GaussianTransientSimulatedSource(
                position=SkyCoord(ra=positions[i][1], dec=positions[i][0]),
                peak_amplitude=peak_amplitudes[i],
                peak_time=peak_times[i],
                flare_width=flare_widths[i],
            )
        elif flare_morphs[i] == "Fixed":
            transient = FixedSimulatedSource(
                position=SkyCoord(ra=positions[i][1], dec=positions[i][0]),
                flux=peak_amplitudes[i],
            )
        else:
            raise ValueError(f"Unsupported flare morphology: {flare_morphs[i]}")

        transients.append(transient)

    return transients
