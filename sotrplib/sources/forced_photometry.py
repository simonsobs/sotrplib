import numpy as np
from typing import List, Union

from pixell import enmap


def enmap_extract_fluxes(flux_map:enmap.ndmap,
                         source_catalog:dict,
                         snr_map:enmap.ndmap=None,
                         dflux_map:enmap.ndmap=None,
                         atmode:str='nn',
                         return_pixels:bool=False,
                         rakey:str='RADeg',
                         deckey:str='decDeg',
                         fluxkey:str='fluxJy',
                         dfluxkey:str='err_fluxJy'
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
    option to return pixel array as well (i.e. the pixel index for the ra,dec of each source)
    '''
    from pixell.utils import degree

    ras = source_catalog.pop(rakey)*degree
    decs = source_catalog.pop(deckey)*degree
    
    if fluxkey in source_catalog:
        fluxes = source_catalog.pop(fluxkey)
    else:
        fluxes = np.zeros(len(ras))*np.nan
    if dfluxkey in source_catalog:
        fluxerrs = source_catalog.pop(dfluxkey)
    else:
        fluxerrs = np.zeros(len(ras))*np.nan
    
    pixels = []
    for i in range(len(ras)):
        pos = [decs[i],ras[i]]
        fluxes[i] = flux_map.at(pos,
                                mode=atmode
                               )
        if isinstance(dflux_map,enmap.ndmap):
            fluxerrs[i] = dflux_map.at(pos,
                                       mode=atmode
                                      )
        elif isinstance(snr_map,enmap.ndmap):
            fluxerrs[i] = fluxes[i]/snr_map.at(pos,
                                               mode=atmode
                                              )

        pixels.append(flux_map.sky2pix(pos))

    source_catalog[fluxkey] = np.array(fluxes)
    source_catalog[dfluxkey] = np.array(fluxerrs)
    source_catalog[rakey] = ras/degree
    source_catalog[deckey] = decs/degree
    
    if return_pixels:
        source_catalog['pix'] = np.array(pixels)
    
    return source_catalog