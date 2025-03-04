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
                            seed:int=None,
                            freq:str='f090',
                            ):
    '''
    
    
    '''
    from photutils.psf import GaussianPSF
    from photutils.datasets import make_model_params, make_model_image, make_noise_image
    from pixell.utils import degree,arcmin
    from pixell.utils import fwhm as fwhm_to_sigma

    ## for matched filtering
    from pixell.analysis import matched_filter_white
    from pixell.uharm import UHT

    #Omega_b is the beam solid angle for a Gaussian beam in sr
    bsigma     = gauss_fwhm_arcmin*fwhm_to_sigma*arcmin
    omega_b    = (np.pi / 4 / np.log(2)) * (2.2*arcmin)**2
    
    model = GaussianPSF()
    shape = sim_map.shape
    mapres = abs(sim_map.wcs.wcs.cdelt[0])*degree
    
    gaussfwhm = gauss_fwhm_arcmin*arcmin/mapres
    minfwhm = gaussfwhm-(fwhm_uncert_frac*gaussfwhm)
    maxfwhm = gaussfwhm+(fwhm_uncert_frac*gaussfwhm)

    if not seed:
        print('No seed provided, setting seed with current ctime')
        seed=np.random.seed()

    if min_flux_Jy>=max_flux_Jy:
        raise ValueError("Min injected flux is less than or equal to max injected flux")
    
    params = make_model_params(shape, 
                               n_sources, 
                               min_separation=5,
                               border_size=5,
                               flux=(min_flux_Jy, max_flux_Jy), 
                               x_fwhm=(minfwhm, maxfwhm),
                               y_fwhm=(minfwhm, maxfwhm), 
                               theta=(gauss_theta_min, gauss_theta_max), 
                               seed=seed
                               )
    ## set the window for each source to be 10x10 arcmin
    ## make the source map and add to input map
    model_window = int(20*arcmin/mapres)
    model_shape = (model_window, model_window)
    data = make_model_image(shape, 
                            model, 
                            params, 
                            model_shape=model_shape,
                            )
    ## photutils distributes the flux over the solid angle
    ## so this map should be the intensity map in Jy/sr
    data/=omega_b

    sim_map+=data

    ## make white noise map with mean zero
    ## add to input map
    ## assume map pixels are square
    #ivar = map_noise_Jy**-2*mapres**2/arcmin**2

    ## Now map is in Jy/sr
    uht = UHT(sim_map.shape, sim_map.wcs)
    # This is the harmonic transform of the beam in 2D
    beam = np.exp(-0.5*uht.l**2*bsigma**2)
    rho, kappa = matched_filter_white(sim_map,
                                      beam, 
                                      enmap.ndmap(np.ones(sim_map.shape),sim_map.wcs)/map_noise_Jy**2,
                                      uht
                                      )
    flux  = rho/kappa
    ## beams per pixel
    flux*=omega_b/mapres**2
    del sim_map,rho,kappa
    
    noise = make_noise_image(shape, 
                             distribution='gaussian', 
                             mean=0,
                             stddev=map_noise_Jy, 
                             seed=seed
                             )
    
    flux += noise

    ## photutils parameter table is in Astropy QTable
    ## convert to the json format we've been using.
    injected_sources = convert_qtable_to_json(params,
                                              imap=flux
                                              )
    return flux, injected_sources


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