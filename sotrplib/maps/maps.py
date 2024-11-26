import numpy as np
from pixell import enmap
import warnings
from typing import Optional



class Depth1Map:
    def __init__(
        self,
        intensity_map: enmap.ndmap,
        inverse_variance_map: enmap.ndmap,
        rho_map: enmap.ndmap,
        kappa_map: enmap.ndmap,
        time_map: enmap.ndmap,
        wafer_name: str,
        freq: str,
        map_ctime: float,
        is_thumbnail: bool = False,
        res: Optional[float]=None
    ):
        self.intensity_map = clean_map(intensity_map, inverse_variance_map)
        self.inverse_variance_map = inverse_variance_map
        self.rho_map = clean_map(rho_map, inverse_variance_map)
        self.kappa_map = clean_map(kappa_map, inverse_variance_map)
        self.time_map = clean_map(time_map, inverse_variance_map)
        self.wafer_name = wafer_name
        self.freq = freq
        self.map_ctime = map_ctime
        self.is_thumbnail = is_thumbnail
        self.res=res
        if not res:
            self.res = np.abs(intensity_map.wcs.wcs.cdelt[0])

    def snr(self):
        return get_snr(self.rho_map, self.kappa_map)

    def flux(self):
        return get_flux(self.rho_map, self.kappa_map)

    def dflux(self):
        return get_dflux(self.kappa_map)

    def mask_map_edge(self, edgecut):
        from .masks import mask_edge
        return mask_edge(self.kappa_map, edgecut)

    def mask_dustgal(self, galmask):
        from .masks import mask_dustgal
        return mask_dustgal(self.kappa_map, galmask)

    def mask_planets(self):
        from .masks import mask_planets
        return mask_planets(self.time_map,self.map_ctime)
    
    def thumbnail(self,
                  ra:float,
                  dec:float,
                  thumbnail_width_arcmin:float=60
                  ):
        ## extract thumbnails around a location and reproject to local flat-sky
        ## return a Depth1Map object with is_thumbnail=True
        intensity_thumbnail = get_thumbnail(self.intensity_map,
                                            ra,
                                            dec,
                                            size_arcmin=thumbnail_width_arcmin
                                            )
        rho_thumbnail = get_thumbnail(self.rho_map,
                                       ra,
                                       dec,
                                       size_arcmin=thumbnail_width_arcmin
                                       )
        kappa_thumbnail = get_thumbnail(self.kappa_map,
                                       ra,
                                       dec,
                                       size_arcmin=thumbnail_width_arcmin
                                       )
        ivar_thumbnail = get_thumbnail(self.inverse_variance_map,
                                       ra,
                                       dec,
                                       size_arcmin=thumbnail_width_arcmin
                                       )
        ctime_thumbnail = get_thumbnail(self.time_map,
                                        ra,
                                        dec,
                                        size_arcmin=thumbnail_width_arcmin
                                       )
        
        return Depth1Map(intensity_thumbnail,
                         ivar_thumbnail,
                         rho_thumbnail,
                         kappa_thumbnail,
                         ctime_thumbnail,
                         self.wafer_name,
                         self.freq,
                         self.map_ctime,
                         is_thumbnail=True
                        )
    
    def perform_matched_filter(self, 
                               ra_can:list[float], 
                               dec_can:list[float], 
                               source:list[str], 
                               sourcemask:Optional["SourceCatalog"]=None, 
                               detection_threshold:float=5.0
                              ):
        from ..filters import renorm_ms
        mf = np.array(len(ra_can) * [False])
        renorm = np.array(len(ra_can) * [False])
        ra_arr = np.array(len(ra_can) * [np.nan])
        dec_arr = np.array(len(ra_can) * [np.nan])
        for i in range(len(ra_can)):
            if source[i]:
                ra_arr[i] = ra_can[i]
                dec_arr[i] = dec_can[i]

            else:
                ra = ra_can[i]
                dec = dec_can[i]
                data_sub = self.matched_filter_submap(ra, 
                                                      dec, 
                                                      sourcemask, 
                                                      size=0.5
                                                     )

                # check if candidate still exists
                ra, dec, pass_matched_filter = self.recalc_source_detection(ra, 
                                                                            dec, 
                                                                            detection_threshold
                                                                            )
                mf[i] = pass_matched_filter

                if pass_matched_filter:
                    # renormalization if matched filter is successful
                    ## should this be data_sub.snr=?
                    snr_sub_renorm = renorm_ms(data_sub.snr, 
                                                data_sub.wafer_name, 
                                                data_sub.freq, 
                                                ra, 
                                                dec, 
                                                sourcemask,
                                                )
                    ra, dec, pass_renorm = data_sub.recalc_source_detection(ra, 
                                                                            dec, 
                                                                            detection_threshold
                                                                            )
                    renorm[i] = pass_renorm

                ra_arr[i] = ra
                dec_arr[i] = dec

        return ra_arr, dec_arr, mf, renorm


    def matched_filter_submap(self, 
                              ra: float, 
                              dec: float, 
                              sourcemask:Optional["SourceCatalog"]=None, 
                              size:float=0.5
                             ):
        """
        calculate rho and kappa maps given a ra and dec

        Args:
            ra: right ascension in deg
            dec: declination in deg
            sourcemask: point sources to mask, default=None
            size: size of submap in deg, default=0.5

        Returns:
            Dept1 submaps 
        """
        from ..filters import matched_filter
        sub = get_submap(self.intensity_map, 
                         ra, 
                         dec, 
                         size
                        )
        sub_inverse_variance = get_submap(self.inverse_variance_map, 
                                          ra, 
                                          dec, 
                                          size
                                         )
        sub_tmap = get_submap(self.time_map, 
                              ra, 
                              dec, 
                              size
                             )
        rho, kappa = matched_filter(sub,
                                    sub_inverse_variance,
                                    self.wafer_name,
                                    self.freq,
                                    ra,
                                    dec,
                                    size_deg=size,
                                    source_cat=sourcemask,
                                    apod_size_arcmin=5,
                                    )

        return Depth1Map(sub,
                        sub_inverse_variance,
                        rho,
                        kappa,
                        sub_tmap,
                        self.wafer_name,
                        self.freq,
                        self.map_ctime,
                        )

    def find_sources(self):
        return extract_sources(self.snr(),self.flux())

    def recalc_source_detection(self, 
                                ra:float, 
                                dec:float, 
                                matching_radius:float = 2.5
                                ):
        """
        returns new ra/dec of source if source is detected in submap, otherwise returns original ra/dec
        Also returns bool of whether source is detected
        New source must be found within matching_radius of the old one.
        """
        from ..utils.utils import angular_separation
        ra_new, dec_new = self.extract_sources()

        if not ra_new:
            return ra, dec, False
        else:
            # calculate separation from original position. Reject sources not within 1 arcmin
            sep = angular_separation(ra, 
                                     dec, 
                                     ra_new, 
                                     dec_new
                                     ) * 60.0  # in arcmin
            # continue if no sources are within 1 arcmin
            if np.all(sep > matching_radius):
                return ra, dec, False
            # only keep source closest to original position
            else:
                ra_new = ra_new[np.argmin(sep)]
                dec_new = dec_new[np.argmin(sep)]

                return ra_new, dec_new, True


def load_maps(map_path:str)->Depth1Map:
    ## map_path should be /file/path/to/obsid_arr_freq_map.fits
    ## only supports I map right now 

    imap = enmap.read_map(map_path, sel=0) # intensity map
    # check if map is all zeros
    if np.all(imap == 0.0) or np.all(np.isnan(imap)):
        print("map is all nan or zeros, skipping")
        return None
    
    path = map_path.split('map.fits')[0]
    ivar = enmap.read_map(path + "ivar.fits") # inverse variance map
    rho = enmap.read_map(path + "rho.fits", sel=0) # whatever rho is, only I
    kappa = enmap.read_map(path + "kappa.fits", sel=0) # whatever kappa is, only I
    time = enmap.read_map(path + "time.fits") # time map

    ## These should be contained in the map metadata in the future
    arr = path.split("/")[-1].split("_")[2]
    freq = path.split("/")[-1].split("_")[3]
    ctime = float(path.split("/")[-1].split("_")[1])

    return Depth1Map(imap, ivar, rho, kappa, time, arr, freq, ctime)


def kappa_clean(kappa: np.ndarray, rho: np.ndarray):
    kappa = np.maximum(kappa, np.max(kappa) * 1e-3)
    kappa[np.where(rho == 0.0)] = 0.0
    return kappa


def clean_map(imap: np.ndarray, inverse_variance: np.ndarray):
    imap[inverse_variance < (np.max(inverse_variance) * 0.01)] = 0
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

def extract_sources(snr:enmap.ndmap,
                    flux:enmap.ndmap,
                    snr_threshold:float = 5.0,
                    verbosity:int = 1,
                   ):
        """
        Detects sources and returns center of mass by flux

        Args:
            snr: signal-to-noise ratio map for detection
            flux: flux map for position
            snr_threshold: signal detection threshold
            mapdata: other maps to include in output
            tags: data to include in every line, such as ctime or map name

        Returns:
            ra and dec of each source in deg

        """
        from scipy import ndimage
        labels, nlabel = ndimage.label(snr > snr_threshold)
        cand_pix = ndimage.center_of_mass(flux, 
                                          labels, 
                                          np.arange(nlabel) + 1
                                         )
        cand_pix = np.array(cand_pix)
        ra = []
        dec = []
        for pix in cand_pix:
            pos = flux.pix2sky(pix)
            d = pos[0] * 180.0 / np.pi
            if d < -90.0:
                d = 360.0 + d
            if d > 90.0:
                d = 360.0 - d
            dec.append(d)
            ra.append(pos[1] * 180.0 / np.pi)
        if len(ra) == 0:
            if verbosity > 0:
                print("did not find any candidates")

        return ra, dec

def get_submap(imap:enmap.ndmap, 
               ra_deg:float, 
               dec_deg:float, 
               size_deg:float=0.5
               )->enmap:
        ## Does not reproject
        from pixell.utils import degree
        ra = ra_deg * degree
        dec = dec_deg * degree
        radius = size_deg * degree
        omap = imap.submap([[dec - radius, ra - radius], 
                            [dec + radius, ra + radius]]
                          )
        return omap


def get_thumbnail(imap:enmap.ndmap, 
                  ra_deg:float, 
                  dec_deg:float, 
                  size_arcmin:float=60
                  )->enmap:
    from pixell import reproject
    from pixell.utils import degree
    ra = ra_deg * degree
    dec = dec_deg * degree
    omap = reproject.thumbnails(imap, 
                                [dec, ra], 
                                size_arcmin * degree/60.
                                )
    return omap

