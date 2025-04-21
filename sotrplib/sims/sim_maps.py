from pixell import enmap
import numpy as np


def make_enmap(center_ra:float=0.0,
               center_dec:float=0.0,
               width_ra:float=1.0,
               width_dec:float=1.0,
               resolution:float=0.5,
               ):
    '''
    
    
    '''
    from pixell.enmap import geometry, zeros
    from pixell.utils import degree, arcmin

    if (width_ra<=0) or (width_dec<=0):
        raise ValueError("Map width must be positive")
    
    min_dec = center_dec-width_dec
    max_dec = center_dec+width_dec
    if min_dec<-90:
        raise ValueError("Minimum declination, %.1f, is below -90"%min_dec)
    if max_dec>90:
        raise ValueError("Maximum declination, %.1f, is above 90"%max_dec)
    
    min_ra = center_ra-width_ra
    max_ra = center_ra+width_ra
    ## need to do something with ra limits unless geometry knows how to do the wrapping.

    box = np.array([[min_dec,min_ra],[max_dec,max_ra]])
    shape, wcs = geometry(box*degree, res=resolution*arcmin)

    return zeros(shape,wcs=wcs)


def photutils_sim_n_sources(sim_map:enmap.ndmap,
                            n_sources:float,
                            min_flux_Jy:float=0.1,
                            max_flux_Jy:float=2.0,
                            map_noise_Jy:float=0.01,
                            gauss_fwhm_arcmin:float=2.2,
                            fwhm_uncert_frac:float=0.3,
                            gauss_theta_min:float=0,
                            gauss_theta_max:float=90,
                            min_sep_arcmin:float=5,
                            seed:int=None,
                            freq:str='f090',
                            ):
    '''
    
    
    '''
    from .sim_utils import convert_photutils_qtable_to_json
    from photutils.psf import GaussianPSF, make_psf_model_image
    from photutils.datasets import make_noise_image
    from pixell.utils import degree,arcmin
    
    model = GaussianPSF()
    shape = sim_map.shape
    wcs = sim_map.wcs
    mapres = abs(wcs.wcs.cdelt[0])*degree
    
    gaussfwhm = gauss_fwhm_arcmin*arcmin
    #Omega_b is the beam solid angle for a Gaussian beam in sr
    omega_b = (np.pi / 4 / np.log(2)) * (gaussfwhm)**2
    
    gaussfwhm/=mapres
    minfwhm = gaussfwhm-(fwhm_uncert_frac*gaussfwhm)
    maxfwhm = gaussfwhm+(fwhm_uncert_frac*gaussfwhm)

    minsep=min_sep_arcmin*arcmin/mapres

    if not seed:
        print('No seed provided, setting seed with current ctime')
        seed=np.random.seed()

    if min_flux_Jy>=max_flux_Jy:
        raise ValueError("Min injected flux is less than or equal to max injected flux")
    
    ## set the window for each source to be 10x10 arcmin
    ## make the source map and add to input map
    model_window = (int(2*minsep),int(2*minsep))
    data,params = make_psf_model_image(shape, 
                                        model, 
                                        n_sources, 
                                        model_shape=model_window,
                                        min_separation=minsep,
                                        border_size=int(minsep),
                                        flux=(min_flux_Jy, max_flux_Jy), 
                                        x_fwhm=(minfwhm, maxfwhm),
                                        y_fwhm=(minfwhm, maxfwhm), 
                                        theta=(gauss_theta_min, gauss_theta_max), 
                                        seed=seed,
                                        normalize=False
                                        )
    ## photutils distributes the flux over the solid angle
    ## so this map should be the intensity map in Jy/sr
    ## multiply by beam area in sq pixels
    sim_map+=data*(omega_b/mapres**2)

    ## make white noise map with mean zero
    ## add to input map
    
    noise = make_noise_image(shape, 
                             distribution='gaussian', 
                             mean=0,
                             stddev=map_noise_Jy, 
                             seed=seed
                             )
    
    sim_map += noise

    ## photutils parameter table is in Astropy QTable
    ## convert to the json format we've been using.
    injected_sources = convert_photutils_qtable_to_json(params,
                                                        imap=sim_map
                                                        )
    return sim_map, injected_sources


def inject_sources(imap:enmap.ndmap, 
                   sources:list, 
                   observation_time:float|enmap.ndmap,
                   freq:str='f090', 
                   arr:str=None, 
                   debug:bool=False
                   ):
    """
    Inject a list of SimTransient objects into the input map.

    Parameters:
    - imap: enmap.ndmap, the input map to inject sources into.
    - sources: list of SimTransient objects.
    - observation_time: float or enmap.ndmap, the time of the observation.
    - freq: str, the frequency of the observation.
    - arr: str, the array name (optional).
    - debug: bool, if True, print debug information including tqdm.

    Returns:
    - imap: enmap.ndmap, the map with sources injected.
    - injected_sources: list of SourceCandidate objects representing the injected sources.
    """
    from pixell.utils import degree, arcmin
    from photutils.psf import GaussianPSF
    from photutils.datasets import make_model_image,make_model_params
    from pixell.enmap import  sky2pix
    from tqdm import tqdm
    from ..utils.utils import get_fwhm
    from ..sources.sources import SourceCandidate

    shape = imap.shape
    wcs = imap.wcs
    mapres = abs(wcs.wcs.cdelt[0]) * degree
    fwhm  = get_fwhm(freq, arr=arr)*arcmin
    fwhm_pixels = fwhm / mapres
    ## calculate this to convert photutils injected flux to Jy
    omega_b = (np.pi / 4 / np.log(2)) * (fwhm_pixels)**2
    injected_sources = []
    removed_sources = {'not_flaring':0,'out_of_bounds':0}
    
    for source in tqdm(sources,desc='Injecting sources',disable=not debug):
        # Check if source is within the map bounds
        ra, dec = source.ra, source.dec
        pix = sky2pix(shape, wcs, np.asarray([dec, ra])*degree)
        
        if not (0 <= pix[0] < shape[-2] and 0 <= pix[1] < shape[-1]):
            removed_sources['out_of_bounds'] += 1
            continue
        mapval = imap[int(pix[0]),int(pix[1])]
        if not np.isfinite(mapval) or abs(mapval)==0:
            removed_sources['out_of_bounds'] += 1
            continue
        
        # Check if the source is flaring at the observation time
        if isinstance(observation_time,float):
            source_obs_time = observation_time
        else:
            source_obs_time = observation_time[int(pix[0]),int(pix[1])]
       
        if abs(source.peak_time - source_obs_time) > 3*source.flare_width:
            removed_sources['not_flaring'] += 1
            continue
        
        flux = source.get_flux(source_obs_time)
        
        # Define the PSF model parameters
        model_params = make_model_params(shape,
                                        1,
                                        flux=(flux,flux),
                                        x_0=(pix[1],pix[1]),
                                        y_0=(pix[0],pix[0]),
                                        x_fwhm=(fwhm_pixels,fwhm_pixels),
                                        y_fwhm=(fwhm_pixels,fwhm_pixels),
                                        theta=(0,0)
                                        )
  
        psf_image= make_model_image(shape,
                                    GaussianPSF(),
                                    model_params,
                                    model_shape=(int(3 * fwhm_pixels), int(3 * fwhm_pixels)),
                                    )*omega_b
        
        imap += psf_image

        inj_source = SourceCandidate(ra=ra,
                                     err_ra=0.0,
                                     dec=dec,
                                     err_dec=0.0,
                                     flux=flux,
                                     err_flux=0.0,
                                     snr=np.inf,
                                     freq=freq,
                                     arr=arr if arr else 'any',
                                     ctime=source_obs_time,
                                     fwhm_a=fwhm/arcmin,
                                     fwhm_b=fwhm/arcmin,
                                     source_type='simulated',
                                    )
        injected_sources.append(inj_source)

    if debug:
        print(f"Removed sources: {removed_sources}")
        print(f"Injected sources: {len(injected_sources)}")

    return imap, injected_sources


