from typing import Optional, List
from ..maps.maps import Depth1Map

def get_depth1_mapset(map_path:str)->List[str]:
    ## assume map_path is the path to the map.fits file
    ## and that the other files are in the same base directory.
    path = map_path.split('map.fits')[0]
    return [map_path,path+'ivar.fits',path+'rho.fits',path+'time.fits']


def load_depth1_mapset(map_path:str,
                       ivar_path:Optional[str]=None,
                       rho_path:Optional[str]=None,
                       kappa_path:Optional[str]=None,
                       time_path:Optional[str]=None,
                       polarization_selector: Optional[int] = 0,
                       )->Depth1Map:
    ## map_path should be /file/path/to/obsid_arr_freq_map.fits
    ## if other paths are none, will replace the .map.fits with , for example ivar.fits
    ##     for the inverse variance.
    ## polarization_selector is 0 for I, 1 for Q, and 2 for U 
    from pixell.enmap import read_map
    import os.path
    
    imap = read_map(map_path, sel=polarization_selector) # intensity map
    # check if map is all zeros
    if np.all(imap == 0.0) or np.all(np.isnan(imap)):
        print("map is all nan or zeros, skipping")
        return None
    
    path = map_path.split('map.fits')[0]
    ivar = read_map(ivar_path) if ivar_path else read_map(path + "ivar.fits") # inverse variance map
    rho = read_map(rho_path, sel=polarization_selector) if rho_path else read_map(path + "rho.fits", sel=polarization_selector) # whatever rho is, only I
    kappa = read_map(kappa_path, sel=polarization_selector) if kappa_path else read_map(path + "kappa.fits", sel=polarization_selector) # whatever kappa is, only I
    time = read_map(time_path) if time_path else read_map(path + "time.fits") # time map

    ## These should be contained in the map metadata in the future
    arr = path.split("/")[-1].split("_")[2]
    freq = path.split("/")[-1].split("_")[3]
    ctime = float(path.split("/")[-1].split("_")[1])

    return Depth1Map(imap, ivar, rho, kappa, time, arr, freq, ctime)