from .sim_utils import generate_random_positions, make_gaussian_flare
import sqlite3
import pickle

class SimTransient:
    def __init__(self, 
                 position=None, 
                 peak_amplitude=0.0, 
                 peak_time=None, 
                 flare_width=None, 
                 flare_morph='Gaussian',
                 beam_params={}
                 ):
        """
        Initialize a simulated source.

        Parameters:
        - position: Tuple (ra, dec) specifying the position of the source. If None, a random position is generated.
        - peak_amplitude: The peak amplitude of the flare.
        - peak_time: The time at which the flare peaks.
        - flare_width: The width of the flare (e.g., standard deviation for Gaussian).
        - flare_morph: The morphology of the flare ('Gaussian' supported for now).
        - beam_params: Dictionary of beam parameters (e.g., FWHM, ellipticity).
        """
        self.ra,self.dec = position if position else generate_random_positions(n=1)
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
        if self.flare_morph == 'Gaussian':
            # Calculate Gaussian amplitude
            if self.peak_time is None or self.flare_width is None:
                raise ValueError("peak_time and flare_width must be defined for Gaussian flare.")
            delta_time = time - self.peak_time
            flux = make_gaussian_flare(delta_time, 
                                       flare_fwhm_s=self.flare_width,
                                       flare_peak_Jy=self.peak_amplitude
                                       )
            return flux
        else:
            raise NotImplementedError(f"Flare morphology '{self.flare_morph}' is not supported.")
        

def generate_transients(n:int=None,
                        imap:enmap.ndmap=None,
                        ra_lims:tuple=None,
                        dec_lims:tuple=None,
                        positions:list=None,
                        peak_amplitudes:list|tuple=None,
                        peak_times:list|tuple=None,
                        flare_widths:list|tuple=None,
                        flare_morphs:list=None,
                        beam_params:list=None,
                        ):
    """
    Generate a list of simulated transient sources.
    Each source is represented by a SimTransient object.
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
    """ 
    from .sim_utils import (generate_random_positions, 
                            generate_random_flare_amplitudes, 
                            generate_random_flare_widths,
                            generate_random_flare_times,
                            )       
            
    transients = []
    if n is None and not positions:
        raise ValueError("Either n or positions must be provided.")
    if n is not None and positions:
        raise ValueError("Cannot provide both n and positions.")
    if n is not None:
        positions = generate_random_positions(n, 
                                              imap=imap,
                                              ra_lims=ra_lims, 
                                              dec_lims=dec_lims
                                              )
        
    if n is not None and isinstance(peak_amplitudes,tuple):
        peak_amplitudes = generate_random_flare_amplitudes(n,
                                                           min_amplitude=peak_amplitudes[0],
                                                           max_amplitude=peak_amplitudes[1]
                                                          )
    elif n is not None and isinstance(peak_amplitudes,type(None)):
        peak_amplitudes = generate_random_flare_amplitudes(n)

    if n is not None and isinstance(peak_times,tuple):
        peak_times = generate_random_flare_times(n,
                                                 start_time=peak_times[0],
                                                 end_time=peak_times[1]
                                                )
    elif n is not None and isinstance(peak_times,type(None)):
        peak_times = generate_random_flare_times(n)

    if n is not None and isinstance(flare_widths,tuple):
        flare_widths = generate_random_flare_widths(n,
                                                     min_width=flare_widths[0],
                                                     max_width=flare_widths[1]
                                                    )
    elif n is not None and isinstance(flare_widths,type(None)):
        flare_widths = generate_random_flare_widths(n)

    if n is not None and isinstance(flare_morphs,type(None)):
        flare_morphs = ['Gaussian'] * n
        
    if n is not None and isinstance(beam_params, type(None)):
        beam_params = [{}] * n
    
    if len(positions) != len(peak_amplitudes) or len(positions) != len(peak_times) or len(positions) != len(flare_widths):
        raise ValueError("All input lists must be of the same length.")
    
    for i in range(len(positions)):
        transient = SimTransient(position=positions[i],
                                 peak_amplitude=peak_amplitudes[i],
                                 peak_time=peak_times[i],
                                 flare_width=flare_widths[i],
                                 flare_morph=flare_morphs[i],
                                 beam_params=beam_params[i]
                                )
        transients.append(transient)
    
    return transients
