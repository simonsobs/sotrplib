from pixell import enmap
import numpy as np

class depth1_lightcurve_extractor:
    def __init__(self,
                 ra:float,
                 dec:float,
                 rho_map:str,
                 kappa_map:str,
                 time_map:str,
                 time:float=None,
                 ):
        self.ra = ra 
        self.dec = dec
        self.time = time
        self.rho_map = rho_map
        self.kappa_map = kappa_map
        self.time_map = time_map
        
        self.source_in_map = None
        self.flux = None
        self.flux_uncert = None
        
    def get_map_info(self):
        ## assumes files named depth1_[ctime]_pa[x]_f[freq]_[suffix]
        name = self.rho_map.split("_rho.fits")[0]
        ttag, arr, ftag = name.split("_")[1:4]
        return ttag,arr,ftag
            
    def is_source_in_map(self):
        from pixell.utils import degree
        shape, wcs = enmap.read_map_geometry(self.rho_map)
        pixs       = enmap.sky2pix(shape, wcs, np.asarray([[self.dec*degree],[self.ra*degree]]))
        inside     = bool(len(np.where(np.all((pixs.T >= 0)&(pixs.T<shape[-2:]),-1))[0]))
        self.source_in_map=inside
        return inside
    
    def get_flux(self):
        from pixell.utils import degree
        
        if self.source_in_map == None:
            self.is_source_in_map()
        if not self.source_in_map:
            self.flux = np.nan
            self.fluxerr = np.nan
            self.time = np.nan
            return 
        poss = np.asarray([[self.dec*degree],[self.ra*degree]])
        rho = enmap.read_map(self.rho_map).at(poss)
        kappa = enmap.read_map(self.kappa_map).at(poss) 
        self.flux = rho/kappa
        self.flux_uncert = kappa**-0.5
        self.time = enmap.read_map(self.time_map).at(poss) 
        return self.flux
    
