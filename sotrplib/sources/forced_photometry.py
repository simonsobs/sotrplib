import numpy as np
from typing import List, Union

from pixell import enmap


def enmap_extract_fluxes(imap:enmap.ndmap,
                         ras:Union[List,np.array],
                         decs:Union[List,np.array],
                         snr_map:enmap.ndmap=None,
                         dflux_map:enmap.ndmap=None,
                         atmode='nn'
                        ):
    '''
    input flux map will have the catalog ra,dec positions extracted using the ndmap.at function
    if dflux_map is input, will use that to get flux uncertainty, otherwise
    if snr_map is input, will use flux/snr to get flux uncertainty

    
    atmode is the interpolation mode used ... from pixell.utils.interpol:
    The mode and order arguments control the interpolation type. These can be:
	* mode=="nn"  or (mode=="spline" and order==0): Nearest neighbor interpolation
	* mode=="lin" or (mode=="spline" and order==1): Linear interpolation
	* mode=="cub" or (mode=="spline" and order==3): Cubic interpolation
	* mode=="fourier": Non-uniform fourier interpolation

    return two arrays, 1st is flux, 2nd is flux uncert
    '''
    forced_phot_flux = []
    forced_phot_fluxerr = []
    for i in range(len(ras)):
        pos = [decs[i],ras[i]]
        fluxi = imap.at(pos,
                        mode=atmode
                        )
        forced_phot_flux.append(fluxi)
        if isinstance(dflux_map,type(enmap.ndmap)):
            dfluxi = dflux_map.at(pos,
                                  mode=atmode
                                  )
            forced_phot_fluxerr.append(dfluxi)
        elif isinstance(snr_map,type(enmap.ndmap)):
            dfluxi = fluxi/snr_map.at(pos,
                                      mode=atmode
                                      )
            forced_phot_fluxerr.append(dfluxi)
        
    return np.array(forced_phot_flux),np.array(forced_phot_fluxerr)