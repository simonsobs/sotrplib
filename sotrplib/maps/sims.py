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