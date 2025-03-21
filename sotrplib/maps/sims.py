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
    injected_sources = convert_qtable_to_json(params,
                                              imap=sim_map
                                              )
    return sim_map, injected_sources


def inject_white_noise(imap:enmap.ndmap,
                       amplitude=10.
                       ):
    from pixell.enmap import rand_gauss
    noise = rand_gauss(imap.shape,imap.wcs)*amplitude

    return imap+noise


def inject_sources(imap:enmap.ndmap,
                   n_sources:int=0,
                   src_fwhm:list=[],
                   src_positions:list=[],
                   src_amplitudes:list=[],
                   src_ellipticity:list=[],
                   src_theta:list=[],
                   position_unit='coord',
                   return_source_catalog=True
                  ):
    '''
    position_units 'coord' or 'pix'
    '''
    from pixell.enmap import modrmap
    from pixell.utils import arcmin,degree
    from pixell.utils import fwhm as FWHM


    if (n_sources>0) and (len(src_positions)==0):
        for i in range(n_sources):
            ## get random positions within imap
            raise NotImplementedError("generating n random sources not yet implemented.")

    for i,pos in enumerate(src_positions):
        bsigma = src_fwhm[i]*FWHM*arcmin
        signal = src_amplitudes[i] * np.exp(-0.5*modrmap(imap.shape, imap.wcs, np.array(pos)*degree)**2/bsigma**2)
        
        imap+=signal
    
    return imap


def get_random_euclidean_positions(imap:enmap.ndmap,
                                   n:int
                                  ):
    from pixell.utils import degree
    
    rand0 = np.random.randint(0,high=imap.shape[0],size=n)
    rand1 = np.random.randint(0,high=imap.shape[1],size=n)
    inj_pix = list(zip(rand0,rand1))
    inj_pos = []
    for pix in inj_pix:
        inj_pos.append(imap.pix2sky(pix)/degree)
    
    return inj_pos


def convert_qtable_to_json(params,
                           imap:enmap.ndmap=None
                           ):
    '''
    photutils params is an Astropy QTable with:
    id,x_0, y_0, flux, fwhm_x,fwhm_y, theta
    
    if imap supplied, will also get ra,dec
    '''
    from pixell.utils import degree
    json_cat_out = {}
    for p in params:
        for k in p.keys():
            if k not in json_cat_out:
                json_cat_out[k] = []
            json_cat_out[k].append(p[k])
        
        if isinstance(imap,enmap.ndmap):
            dec,ra=imap.pix2sky([p['y_0'],p['x_0']])/degree
        else:
            ra,dec = np.nan,np.nan
        if 'RADeg' not in json_cat_out:
            json_cat_out['RADeg'] = []
            json_cat_out['decDeg'] = []
            json_cat_out['name'] = []
            
        json_cat_out['RADeg'].append(ra)
        json_cat_out['decDeg'].append(dec)
        json_cat_out['name'].append(str(p['id']))
    for key in json_cat_out:
        json_cat_out[key] = np.array(json_cat_out[key])
    flux = json_cat_out.pop('flux')
    json_cat_out['fluxJy'] = flux
    return json_cat_out