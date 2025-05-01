import numpy as np

from .maps import Depth1Map


def coadd_map_group(map_group:list,
                    map_res:float=None,
                    freq:str=None,
                    arr:str=None,
                    box:np.ndarray=None,
                   ):
    from tqdm import tqdm
    from pixell.enmap import read_map_geometry
    from pixell.utils import degree
    if not map_res:
        map_res = abs(read_map_geometry(str(map_group[0]))[1].wcs.cdelt[0])*degree
    coadd=Depth1Map()
    coadd.init_empty_so_map(res=map_res,box=box)
    coadd.box=box
    if not coadd.freq:
        coadd.freq = freq
    if not coadd.wafer_name:
        coadd.wafer_name = arr
    
    for mg in tqdm(range(len(map_group))):
        map_path = map_group[mg]
        coadd.coadd_maps(map_path,box=box)
    coadd.time_map/=coadd.kappa_map
    coadd.map_ctime = np.nanmean(coadd.time_map)
    
    return coadd


def write_coadd(coadd,
                output_dir:str,
                outfile_prefix:str,
                extra_info:dict={},
                maptypes:list=["rho_map","kappa_map","time_map","flux_map","intensity_map","ivar_map"]
                ):
    '''
    coadd should be a Depth1Map object
    outfile_prefix should be something like coadd_[map_ctime]_all_[freq] 
    where all is the stand-in for array (to keep same notation as depth1 maps)
    
    extra_info is the extra info written to the fits header; should be compatible with that,
    for example:
    extra_info={'obsids':','.join(coadd.maps_included),
            'cleaned':False,
            'freq':freq,
            'ndays':coadd_days,
        }
    '''
    from pixell.enmap import write_map

    for mt in maptypes:
        if not isinstance(getattr(coadd,mt),type(None)):
            write_map(output_dir+'%s_%s.fits'%(outfile_prefix,mt),
                      getattr(coadd,mt),
                      extra=extra_info
                     )
    return

def load_coadd_maps(mapfile:str=None,
                    map_dir:str=None,
                    ctime:float=None,
                    debug:bool=False,
                    arr:str=None,
                    box:np.ndarray=None,
                   ):
    from pathlib import Path
    from glob import glob
    
    if not isinstance(mapfile,type(None)):
        maps = [mapfile]
    elif isinstance(map_dir,type(None)):
        print('One of mapfile, map_dir needs to be input.')
        return []
    else:
        if ctime:
            globstr = map_dir+'*_%s_*rho.fits'%ctime
        else:
            globstr = map_dir+'*_rho.fits'
        if debug:
            print(globstr)
        maps = glob(globstr)

    coadds = []
    for m in maps:
        freq = 'f'+m.split('_f')[-1].split('_')[0]
        if debug:
            print(m,freq)
        mapstub = Depth1Map()
        mapstub.load_coadd(Path(m),box=box)
        if arr:
            mapstub.wafer_name=arr
        coadds.append(mapstub)

    return coadds