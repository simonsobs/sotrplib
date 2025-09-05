from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pixell import enmap


class SimTransient:
    def __init__(
        self,
        position: tuple[AstroPydanticQuantity[u.deg], AstroPydanticQuantity[u.deg]]
        | list[AstroPydanticQuantity[u.deg]] = None,
        peak_amplitude: AstroPydanticQuantity[u.Jy] = 0.0 * u.Jy,
        peak_time: float = None,
        flare_width: float = None,
        flare_morph: str = "Gaussian",
        beam_params: dict = {},
    ):
        """
        Initialize a simulated source.

        Parameters:
        - position: Tuple|list (dec, ra) specifying the position of the source. If None, a random position is generated.
        - peak_amplitude: The peak amplitude of the flare.
        - peak_time: The time at which the flare peaks.
        - flare_width: The width of the flare (e.g., standard deviation for Gaussian).
        - flare_morph: The morphology of the flare ('Gaussian' supported for now).
        - beam_params: Dictionary of beam parameters (e.g., FWHM, ellipticity).
        """
        if position is None:
            from sotrplib.sims.sim_utils import generate_random_positions

            self.dec, self.ra = generate_random_positions(
                n=1, ra_lims=(0, 360) * u.deg, dec_lims=(-60, 20) * u.deg
            )[0]
            self.dec *= u.deg
            self.ra *= u.deg
        else:
            self.dec, self.ra = position

        self.peak_amplitude = peak_amplitude
        self.peak_time = peak_time
        self.flare_width = flare_width
        self.flare_morph = flare_morph
        self.beam_params = beam_params

    def get_flux(self, time):
        """
        Calculate the flux (amplitude) at a given time based on the flare morphology.

        Parameters:
        - time: The time at which to evaluate the flux.

        Returns:
        - Flux (amplitude) value at the given time.
        """
        from sotrplib.sims.sim_utils import make_gaussian_flare

        if self.flare_morph == "Gaussian":
            # Calculate Gaussian amplitude
            if self.peak_time is None or self.flare_width is None:
                raise ValueError(
                    "peak_time and flare_width must be defined for Gaussian flare."
                )
            delta_time = time - self.peak_time
            flux = make_gaussian_flare(
                delta_time,
                flare_fwhm_s=self.flare_width,
                flare_peak_Jy=self.peak_amplitude.to(u.Jy).value,
            )
            return flux * u.Jy
        else:
            raise NotImplementedError(
                f"Flare morphology '{self.flare_morph}' is not supported."
            )


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
    log=None,
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
        transient = SimTransient(
            position=(positions[i][0], positions[i][1]),
            peak_amplitude=peak_amplitudes[i],
            peak_time=peak_times[i],
            flare_width=flare_widths[i],
            flare_morph=flare_morphs[i],
            beam_params=beam_params[i],
        )
        transients.append(transient)

    return transients
