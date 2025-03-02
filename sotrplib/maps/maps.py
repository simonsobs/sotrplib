import numpy as np
import warnings
from typing import Optional
from pathlib import Path


import numpy as np
from pixell import enmap
from pixell.utils import arcmin,degree


class Depth1Map:
    def __init__(
        self,
        intensity_map: Path=None,
        inverse_variance_map:Path=None,
        rho_map: Path=None,
        kappa_map: Path=None,
        time_map: Path=None,
        wafer_name: str=None,
        freq: str=None,
        map_ctime: float=-1.,
        is_thumbnail: bool = False,
        res:float=None,
        map_start_time:float=None,
        units:str=None
    ):
        self.inverse_variance_map = inverse_variance_map
        self.intensity_map = intensity_map
        self.rho_map = rho_map
        self.kappa_map = kappa_map
        self.time_map = time_map
        self.flux = None
        self.snr = None
        self.units = units

        self.wafer_name = wafer_name
        self.freq = freq
        self.map_ctime = map_ctime
        if not self.wafer_name or not self.freq or not self.map_ctime:
            self.get_map_info(self.rho_map)

        self.is_thumbnail = is_thumbnail
        self.res=res
        self.maps_included = []
        self.cleaned=False
        self.flatfielded = False
        self.masked=False
        self.start_ctime = None
        self.end_ctime = None
        self.coadd_days = None
        self.map_start_time = map_start_time

    def get_map_info(self,map_path):
        if not isinstance(map_path,Path):
            return
        
        suffix = str(map_path).split('/')[-1]
        self.wafer_name = suffix.split("_")[2]
        self.freq = str(map_path).split("/")[-1].split("_")[3]
        self.map_ctime = float(str(map_path).split("/")[-1].split("_")[1])
        self.start_ctime = get_observation_start_time(map_path)


    def init_empty_so_map(self,
                          res:float=None,
                          intensity:bool=False,
                          include_half_pixel_offset=False
                         ):
        ## res in radian
        if not res:
            if not self.res:
                raise Exception("Need a resolution, no input res and self.res is None.")
            res = self.res/arcmin
        elif not self.res:
            self.res = res
        empty_map = enmap.zeros(*widefield_geometry(res=self.res,
                                                    include_half_pixel_offset=include_half_pixel_offset
                                                    )
                               )
        if intensity:
            self.inverse_variance_map = empty_map.copy()
            self.intensity_map = empty_map.copy()
        else:
            self.rho_map = empty_map.copy()
            self.kappa_map = empty_map.copy()
        self.time_map = empty_map.copy()
        

    def load_map(self, map):
        """
        Check if map is a file path and loads it dynamically if so.

        Parameters:
            map: The map or path to map.
        """
        # Define substrings and their corresponding attribute names
        substr_to_attr = {
            "rho": "rho_map",
            "kappa": "kappa_map",
            "time": "time_map",
            "flux": "flux_map",
            "intensity": "intensity_map",
            "ivar": "ivar_map",
        }
        if not isinstance(map,Path):
            return
        
        # Check which attribute the filename corresponds to
        for substr, attr_name in substr_to_attr.items():
            if substr in str(map):
                # load the map and update the attribute
                try:
                    m = enmap.read_map(str(map),sel=0)
                except Exception as e1:
                    try:
                        m = enmap.read_map(str(map))
                    except Exception as e2:
                        print('Failed initially via ',e1)
                        print('Then tried without sel=0, and failed via ',e2)
                        return False
                
                setattr(self, attr_name,m)
                if not self.res:
                    self.res = np.abs(self.rho_map.wcs.wcs.cdelt[0])*degree
                self.get_map_info(map)
                return

    def load_coadd(self,
                   map_path:Path
                   ):
        ## input map should be /file/path/to/obsid_arr_freq_[rho/kappa].fits
        ## only supports rho,kappa,time maps right now 
        ## must be Path object
        from os.path import exists
        if not isinstance(map_path,Path):
            raise TypeError("map_path must be a Path object.")
        if 'rho' in str(map_path):
            maptype='rho'
        elif 'kappa' in str(map_path):
            maptype='kappa'
        ## could do something with intensity and inverse variance maps
        ## or even just intensity map + calculated inverse variance...
        elif 'map' in str(map_path):
            maptype='map'
        elif 'ivar' in str(map_path):
            maptype='ivar'    

        path2map = str(map_path).split(f'{maptype}.fits')[0]
        if not exists(path2map + "rho.fits") or not exists(path2map + "kappa.fits"):
            raise FileNotFoundError(f"One of {path2map}[rho/kappa].fits not found! Cant load.")

        if not exists(path2map + "time.fits"):
            from glob import glob
            timefiles = glob(path2map+'*time.fits')
            if len(timefiles) == 0:
                print(str(path2map)+'*time.fits')
                raise FileNotFoundError("No time map found!")
            elif 'weighted' in timefiles[0]:
                time_map_file = timefiles[0]
                weighted_time=True
        else:
            time_map_file = path2map + "time.fits"
            weighted_time = False
        
        self.rho_map = enmap.read_map(path2map + "rho.fits") # whatever rho is, only I
        self.kappa_map = enmap.read_map(path2map + "kappa.fits") # whatever kappa is, only I
        self.time_map=enmap.read_map(time_map_file)
        if weighted_time:
            self.time_map/=self.kappa_map
        
        ## load in extra info, like obs in map, freq, and Ndays coadded.
        self.load_coadd_info(map_path)
        
        
        return

    def load_coadd_info(self,
                        map_path:Path
                        ):
        import astropy.io.fits as fits
        h = fits.open(map_path)
        try:
            obsids = [float(o.split('_')[0]) for o in h[0].header['OBSIDS'].split(',')]
            self.maps_included = obsids
        except Exception as e:
            print(e,'Failed to read OBSIDS')
        
        try:
            self.coadd_days  = float(h[0].header['NDAYS'])
        except Exception as e:
            print(e,'Failed to read NDAYS')
        
        try:
            self.freq  = h[0].header['FREQ']
        except Exception as e:
            print(e,'Failed to read FREQ')
            self.freq = str(map_path).split('/')[-1].split('_')[3]

        if not self.res:
            self.res = np.abs(self.rho_map.wcs.wcs.cdelt[0])*degree
        del h

        return


    def coadd_maps(self,second_map):
        ## input map should be /file/path/to/obsid_arr_freq_[rho/kappa].fits
        ## only supports I map right now 
        ## must be Path object
        from os.path import exists
        if not isinstance(second_map,Path):
            raise TypeError("second_map must be a Path object.")
        if 'rho' in str(second_map):
            maptype='rho'
        elif 'kappa' in str(second_map):
            maptype='kappa'
        ## could do something with intensity and inverse variance maps
        ## or even just intensity map + calculated inverse variance...
        elif 'map' in str(second_map):
            maptype='map'
        elif 'ivar' in str(second_map):
            maptype='ivar'    

        if  isinstance(self.rho_map,type(None)):
            if not self.res:
                self.res = np.abs(enmap.read_map_geometry(str(second_map))[1].wcs.cdelt[0])*degree
            self.init_empty_so_map()
            self.get_map_info(second_map)

        path2map = str(second_map).split(f'{maptype}.fits')[0]
        if not exists(path2map + "rho.fits") or not exists(path2map + "kappa.fits") or not exists(path2map + "time.fits"):
            raise FileNotFoundError(f"One of {path2map} [rho/kappa/time].fits not found! Cant coadd.")

        arr2 = path2map.split("/")[-1].split("_")[2]
        freq2 = path2map.split("/")[-1].split("_")[3]
        ctime2 = float(path2map.split("/")[-1].split("_")[1])

        second_rho = enmap.read_map(path2map + "rho.fits", sel=0) # whatever rho is, only I
        self.load_map(self.rho_map)
        self.rho_map.insert(second_rho,iwcs=second_rho.wcs,op=lambda a,b:a+b)
        del second_rho

        second_kappa = enmap.read_map(path2map + "kappa.fits", sel=0) # whatever kappa is, only I
        self.load_map(self.kappa_map)
        self.kappa_map.insert(second_kappa,iwcs=second_kappa.wcs,op=lambda a,b:a+b)
        
        second_time = enmap.read_map(path2map + "time.fits")
        t0 = get_observation_start_time(second_map)
        if not t0:
            t0=0.0
        self.time_map.insert((second_time+t0)*second_kappa,iwcs=second_time.wcs,op=lambda a,b:a+b)
        del second_time,second_kappa

        if isinstance(self.maps_included,bool):
            mapinfo = f'{self.map_ctime}_{self.wafer_name}_{self.freq}'
            self.maps_included=[mapinfo]
        self.maps_included.append(f'{ctime2}_{arr2}_{freq2}')

        return
    
def enmap_map_union(map1,map2):
    ## from enmap since enmap.zeros causes it to fail.
    oshape, owcs = enmap.union_geometry([map1.geometry, map2.geometry])
    omap = enmap.zeros(map1.shape[:-2]+oshape[-2:], owcs, map1.dtype)
    omap.insert(map1)
    omap.insert(map2, op=lambda a,b:a+b)
    return omap

def load_maps(map_path:Path)->Depth1Map:
    ## map_path should be /file/path/to/[obsid]_[arr]_[freq]_map.fits
    ## or /file/path/to/[obsid]_[arr]_[freq]_rho.fits
    ## if rho, then loads rho, kappa, time
    ## if map, then loads map, ivar, time
    if 'map.fits' in str(map_path):
        try:
            imap = enmap.read_map(str(map_path), sel=0) # intensity map
        except exception:
            imap = enmap.read_map(str(map_path))
        # check if map is all zeros
        if np.all(imap == 0.0) or np.all(np.isnan(imap)):
            print("map is all nan or zeros, skipping")
            return None
        path = str(map_path).split('map.fits')[0]
        ivar = enmap.read_map(path + "ivar.fits") # inverse variance map
        time = enmap.read_map(path + "time.fits") # time map
        ## These should be contained in the map metadata in the future
        t0=get_observation_start_time(map_path)
        arr = path.split("/")[-1].split("_")[2]
        freq = path.split("/")[-1].split("_")[3]
        ctime = float(path.split("/")[-1].split("_")[1])
        res = np.abs(imap.wcs.wcs.cdelt[0])*degree
        return Depth1Map(intensity_map=imap, 
                         inverse_variance_map=ivar, 
                         time_map=time,
                         wafer_name=arr,
                         freq=freq,
                         map_ctime=ctime,
                         res=res,
                         map_start_time=t0
                        )
    elif 'rho.fits' in str(map_path):
        try:
            rho = enmap.read_map(str(map_path), sel=0) # intensity map
        except Exception:
            rho = enmap.read_map(str(map_path))
        # check if map is all zeros
        if np.all(rho == 0.0) or np.all(np.isnan(rho)):
            print("rho is all nan or zeros, skipping")
            return None
        path = str(map_path).split('rho.fits')[0]
        try:
            kappa = enmap.read_map(path + "kappa.fits", sel=0) # whatever kappa is, only I
        except Exception:
            kappa = enmap.read_map(path + "kappa.fits")
        try:
            time = enmap.read_map(path + "time.fits",sel=0) # time map
        except Exception:
            time = enmap.read_map(path + "time.fits")
        ## These should be contained in the map metadata in the future
        arr = path.split("/")[-1].split("_")[2]
        freq = path.split("/")[-1].split("_")[3]
        ctime = float(path.split("/")[-1].split("_")[1])
        res = np.abs(rho.wcs.wcs.cdelt[0])*degree
        t0=get_observation_start_time(map_path)
        return Depth1Map(rho_map=rho, 
                         kappa_map=kappa, 
                         time_map=time,
                         wafer_name=arr,
                         freq=freq,
                         map_ctime=ctime,
                         res=res,
                         map_start_time=t0
                        )
    

def kappa_clean(kappa: np.ndarray, 
                rho: np.ndarray
                ):
    kappa = np.maximum(kappa, np.nanmax(kappa) * 1e-3)
    kappa[np.where(rho == 0.0)] = 0.0
    return kappa


def clean_map(imap: np.ndarray, 
              inverse_variance: np.ndarray,
              fraction:float=0.01,
              cut_on:str='max'
              ):
    ## cut_on can be max or median, this sets the imap to zero for values of inverse variance
    ## which are below fraction*max or fraction*median of inverse variance map.
    if cut_on=='median' or cut_on=='med':
        imap[inverse_variance < (np.nanmedian(inverse_variance) * fraction)] = 0
    else:
        if cut_on!='max':
            print('%s cut_on not supported, defaulting to max cut'%cut_on)
        imap[inverse_variance < (np.nanmax(inverse_variance) * fraction)] = 0
    return imap


def get_snr(rho: np.ndarray, kappa: np.ndarray):
    # supress divide by zero warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snr = rho / kappa**0.5
    snr[np.where(kappa == 0.0)] = 0.0
    return snr


def get_flux(rho: np.ndarray, kappa: np.ndarray):
    # supress divide by zero warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux = rho / kappa
    ## kind of want to do something else with this -AF
    flux[np.where(kappa == 0.0)] = 0.0
    return flux


def get_dflux(kappa: np.ndarray):
    # supress divide by zero warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dflux = kappa**-0.5
    dflux[np.where(kappa == 0.0)] = 0.0
    return dflux


def get_maptype(map_path:Path):
    if 'rho' in str(map_path):
        maptype='rho'
    elif 'kappa' in str(map_path):
        maptype='kappa'
    elif 'map' in str(map_path):
        maptype='map'
    elif 'ivar' in str(map_path):
        maptype='ivar'
    elif 'time' in str(map_path):
        maptype='time'
    if isinstance(map_path,str):
        suffix = '.'+map_path.split('.')[-1]
    else:
        suffix = map_path.suffix
    return maptype,suffix

def get_observation_start_time(map_path:Path
                              ):
    from os.path import exists
    maptype,_ = get_maptype(map_path)
    infofile  = str(map_path).split(maptype)[0]+"info.hdf"
    if not exists(infofile):
        t0 = None
    else:
        from pixell.bunch import read as bunch_read
        info = bunch_read(infofile)
        t0 = info.t
    return t0


def edge_map(imap: enmap.ndmap):
    """Finds the edges of a map

    Args:
        imap: ndmap to find edges of

    Returns:
        binary ndmap with 1 inside region, 0 outside
    """
    from scipy.ndimage import binary_fill_holes

    edge = enmap.enmap(imap, imap.wcs)  # Create map geometry
    edge[np.abs(edge) > 0] = 1  # Convert to binary
    edge = binary_fill_holes(edge)  # Fill holes

    return enmap.enmap(edge.astype("ubyte"), imap.wcs)


def get_submap(imap:enmap.ndmap, 
               ra_deg:float, 
               dec_deg:float, 
               size_deg:float=0.5,
               )->enmap:
        ## Does not reproject
        from pixell.utils import degree
        ra = ra_deg * degree
        dec = dec_deg * degree
        radius = size_deg * degree
        omap = imap.submap([[dec - radius, ra - radius], 
                            [dec + radius, ra + radius]],
                          )
        return omap


def get_thumbnail(imap:enmap.ndmap, 
                  ra_deg:float, 
                  dec_deg:float, 
                  size_deg:float=0.5,
                  proj:str='tan',
                  )->enmap:
    from pixell import reproject
    from pixell.utils import degree

    ra = ra_deg * degree
    dec = dec_deg * degree
    omap = reproject.thumbnails(imap, 
                                [dec, ra], 
                                size_deg * degree,
                                proj=proj,
                                )
    if np.all(np.isnan(omap)):
        omap = reproject.thumbnails(np.nan_to_num(imap), 
                                    [dec, ra], 
                                    size_deg * degree,
                                    proj=proj,
                                    )
    return omap


def get_time_safe(time_map:enmap.ndmap, 
                  poss:list, 
                  r:float=5.0
                  ):
    ## pos [[dec,ra],[dec,ra],...]
    ## r radius in arcmin
    from pixell.utils import arcmin
    r*=arcmin
    poss     = np.array(poss)
    vals     = time_map.at(poss, order=0)
    bad      = np.where(vals==0)[0]
    if len(bad) > 0:
        pixboxes = enmap.neighborhood_pixboxes(time_map.shape, time_map.wcs, poss.T[bad], r=r)
        for i, pixbox in enumerate(pixboxes):
            thumb = time_map.extract_pixbox(pixbox)
            mask  = thumb != 0
            vals[bad[i]] = np.sum(mask*thumb)/np.sum(mask)
    return vals

def widefield_geometry(res=None, 
                       shape=None, 
                       dims=(), 
                       proj="car", 
                       variant="fejer1",
                       include_half_pixel_offset=False
                      ):
    """Build an enmap covering the full sky, with the outermost pixel centers
	at the poles and wrap-around points. Only the car projection is
	supported for now, but the variants CC and fejer1 can be selected using
	the variant keyword. This currently defaults to CC, but will likely
    change to fejer1 in the future.
    """
    from pixell import wcsutils
    from pixell import utils as pixell_utils
    if variant.lower() == "cc":
        yo = 1
    elif variant.lower() == "fejer1":
        yo = 0
    else:
        raise ValueError("Unrecognized CAR variant '%s'" % str(variant))
    
    # Set up the shape/resolution
    ra_width = 2*np.pi
    dec_width = 80*np.pi/180. ## -60 to +20
    ra_cent = 0
    dec_cent = -20
    if shape is None:
        res   = np.zeros(2)+res
        shape = pixell_utils.nint(([dec_width,ra_width]/res) + (yo,0))
    else:
        res = np.array([dec_width,ra_width])/(np.array(shape)-(yo,0))
    ny, nx = shape
    assert abs(res[0]*(ny-yo) - dec_width) < 1e-8, "Vertical resolution does not evenly divide the sky; this is required for SHTs."
    assert abs(res[1]*nx - ra_width) < 1e-8, "Horizontal resolution does not evenly divide the sky; this is required for SHTs."
    wcs   = wcsutils.WCS(naxis=2)
	# Note the reference point is shifted by half a pixel to keep
	# the grid in bounds, from ra=180+cdelt/2 to ra=-180+cdelt/2.
    #
    wcs.wcs.crval = [0,0]
    if include_half_pixel_offset:
        wcs.wcs.crval = [res[1]/2/pixell_utils.degree,0]
    wcs.wcs.cdelt = [-ra_width/pixell_utils.degree/nx,dec_width/pixell_utils.degree/(ny-yo)]
    wcs.wcs.crpix = [nx//2,(ny-dec_cent*pixell_utils.degree/res[0])/2]#
    if include_half_pixel_offset:
        wcs.wcs.crpix = [nx//2+0.5,(ny+1)/2]#
    wcs.wcs.ctype = ["RA---CAR","DEC--CAR"]
    return dims+(ny,nx), wcs


def preprocess_map(mapdata,
                   galmask_file='/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/mask_for_sources2019_plus_dust.fits',
                   plot_output_dir='./',
                   tilegrid=1.0,
                   PLOT=False):
    ## tilegrid in deg, used for setting median ratio flatfielding grid size
    from pixell.enmap import read_map, extract
    from .masks import mask_edge
    from .tiles import get_medrat,get_tmap_tiles
    if PLOT:
        import matplotlib as mpl
        mpl.use('agg')
        from matplotlib import pylab as plt

    print('Cleaning maps...')
    if not mapdata.cleaned:
        ## ignore divide by zero warning since that will happen outside the weighted region
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mapdata.kappa_map  = kappa_clean(mapdata.kappa_map,
                                            mapdata.rho_map
                                            )
            mapdata.rho_map = clean_map(mapdata.rho_map,
                                        mapdata.kappa_map,
                                        cut_on='median',
                                        fraction=0.05
                                    )
        mapdata.cleaned=True
    ## ignore divide by zero warning since that will happen outside the weighted region
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mapdata.flux = mapdata.rho_map/mapdata.kappa_map
        mapdata.snr = mapdata.rho_map*mapdata.kappa_map**(-0.5)
    mapdata.kappa_map = None
    mapdata.rho_map = None

    if not mapdata.masked:
        mapdata.masked=0
        print('Masking maps...')
        try:
            galaxy_mask = read_map(galmask_file)
            gal_mask = extract(galaxy_mask,mapdata.flux.shape,mapdata.flux.wcs)
            mapdata.flux *= gal_mask
            mapdata.snr *= gal_mask
            del gal_mask,galaxy_mask
            
        except Exception as e:
            print(e,' ... skipping galaxy mask')
            mapdata.masked+=1
        
        try:
            ## could add in planet mask here:
            planet_mask = None
            if planet_mask:
                mapdata.flux *= planet_mask
                mapdata.snr *= planet_mask
            del planet_mask
        except Exception as e:
            print(e,' ... skipping planet mask')
            mapdata.masked+=1

        try:
            edge_mask = mask_edge(mapdata.flux,20)
            mapdata.flux *= edge_mask
            mapdata.snr *= edge_mask
            del edge_mask
        except Exception as e:
            print(e, ' ... skipping edge mask')
            mapdata.masked+=1

    if not mapdata.flatfielded:
        print('Flatfielding maps...')
        med_ratio=None
        try:
            med_ratio = get_medrat(mapdata.snr,
                                   get_tmap_tiles(np.nan_to_num(mapdata.time_map)-np.nanmin(mapdata.time_map),
                                                  tilegrid,
                                                  mapdata.snr,
                                                  id=f"{mapdata.freq}",
                                                 ),
                                  )
            
            #mapdata.flux*=med_ratio
            mapdata.snr*=med_ratio
            mapdata.flatfielded=True
        except Exception as e:
            print(e,' Cannot flatfield at this time... need to update medrat algorithm for coadds')
        if PLOT:
            print('Plotting flatfield...')
            plt.figure(dpi=200)
            plt.imshow(med_ratio,vmax=1,vmin=0)
            plt.colorbar()
            plt.savefig(plot_output_dir+'%i_medratio_map.png'%mapdata.map_ctime)
            plt.close()
        
            plt.figure(dpi=200)
            plt.hist(np.ndarray.flatten(med_ratio[med_ratio>0]),bins=101)
            plt.grid()
            plt.yscale('log')
            plt.savefig(plot_output_dir+'%i_medratio_hist.png'%mapdata.map_ctime)
            plt.close('all')
            
        del med_ratio

    return 
