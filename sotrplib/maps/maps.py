import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from pixell import enmap
from pixell.utils import arcmin, degree


class Depth1Map:
    def __init__(
        self,
        intensity_map: Path = None,
        inverse_variance_map: Path = None,
        rho_map: Path = None,
        kappa_map: Path = None,
        time_map: Path = None,
        wafer_name: str = None,
        freq: str = None,
        map_ctime: float = -1.0,
        is_thumbnail: bool = False,
        res: float = None,
        map_start_time: float = None,
        units: str = None,
        box: np.ndarray = None,
        map_id: str = None,
    ):
        self.inverse_variance_map = inverse_variance_map
        self.intensity_map = intensity_map
        self.rho_map = rho_map
        self.kappa_map = kappa_map
        self.time_map = time_map
        self.flux = None
        self.snr = None
        self.units = units
        self.box = box
        self.wafer_name = wafer_name
        self.freq = freq
        self.map_ctime = map_ctime
        if not self.wafer_name or not self.freq or not self.map_ctime:
            if self.rho_map is not None:
                self.get_map_info(self.rho_map)

        self.is_thumbnail = is_thumbnail
        self.res = res
        self.maps_included = []
        self.cleaned = False
        self.flatfielded = False
        self.masked = False
        self.start_ctime = None
        self.end_ctime = None
        self.coadd_days = None
        self.map_start_time = map_start_time
        self.map_id = map_id

    def get_map_info(self, map_path):
        if not isinstance(map_path, Path):
            return

        suffix = str(map_path).split("/")[-1]
        self.wafer_name = suffix.split("_")[2]
        self.freq = str(map_path).split("/")[-1].split("_")[3]
        self.map_ctime = float(str(map_path).split("/")[-1].split("_")[1])
        self.start_ctime = get_observation_start_time(map_path)

    def init_empty_so_map(
        self,
        res: float = None,
        intensity: bool = False,
        include_half_pixel_offset=False,
        box: np.ndarray = None,
    ):
        ## res in radian
        if not res:
            if not self.res:
                raise Exception("Need a resolution, no input res and self.res is None.")
            res = self.res / arcmin
        elif not self.res:
            self.res = res
        if box is None:
            empty_map = enmap.zeros(
                *widefield_geometry(
                    res=self.res, include_half_pixel_offset=include_half_pixel_offset
                )
            )
        else:
            shape, wcs = enmap.geometry(box, res=self.res)
            empty_map = enmap.zeros(shape, wcs=wcs)

        if intensity:
            self.inverse_variance_map = empty_map.copy()
            self.intensity_map = empty_map.copy()
        else:
            self.rho_map = empty_map.copy()
            self.kappa_map = empty_map.copy()
        self.time_map = empty_map.copy()

    def load_map(self, map, box: np.ndarray = None):
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
        if not isinstance(map, Path):
            return

        # Check which attribute the filename corresponds to
        for substr, attr_name in substr_to_attr.items():
            if substr in str(map):
                # load the map and update the attribute
                try:
                    m = enmap.read_map(str(map), sel=0, box=box)
                except Exception as e1:
                    try:
                        m = enmap.read_map(str(map), box=box)
                    except Exception as e2:
                        print("Failed initially via ", e1)
                        print("Then tried without sel=0, and failed via ", e2)
                        return False

                setattr(self, attr_name, m)
                if not self.res:
                    self.res = np.abs(self.rho_map.wcs.wcs.cdelt[0]) * degree
                self.get_map_info(map)
                self.box = box
                return

    def load_coadd(
        self,
        map_path: Path,
        box: np.ndarray = None,
    ):
        ## input map should be /file/path/to/obsid_arr_freq_[rho/kappa].fits
        ## only supports rho,kappa,time maps right now
        ## must be Path object
        from os.path import exists

        if not isinstance(map_path, Path):
            raise TypeError("map_path must be a Path object.")
        if "rho" in str(map_path):
            maptype = "rho"
        elif "kappa" in str(map_path):
            maptype = "kappa"
        ## could do something with intensity and inverse variance maps
        ## or even just intensity map + calculated inverse variance...
        elif "map" in str(map_path):
            maptype = "map"
        elif "ivar" in str(map_path):
            maptype = "ivar"

        path2map = str(map_path).split(f"{maptype}.fits")[0]
        if not exists(path2map + "rho.fits") or not exists(path2map + "kappa.fits"):
            raise FileNotFoundError(
                f"One of {path2map}[rho/kappa].fits not found! Cant load."
            )

        if not exists(path2map + "time.fits"):
            from glob import glob

            timefiles = glob(path2map + "*time.fits")
            if len(timefiles) == 0:
                print(str(path2map) + "*time.fits")
                raise FileNotFoundError("No time map found!")
            elif "weighted" in timefiles[0]:
                time_map_file = timefiles[0]
                weighted_time = True
        else:
            time_map_file = path2map + "time.fits"
            weighted_time = False

        self.rho_map = enmap.read_map(
            path2map + "rho.fits", box=box
        )  # whatever rho is, only I
        self.kappa_map = enmap.read_map(
            path2map + "kappa.fits", box=box
        )  # whatever kappa is, only I
        self.time_map = enmap.read_map(time_map_file, box=box)  # time map
        if weighted_time:
            self.time_map /= self.kappa_map

        ## load in extra info, like obs in map, freq, and Ndays coadded.
        self.load_coadd_info(map_path)
        self.box = box

        return

    def load_coadd_info(self, map_path: Path):
        import astropy.io.fits as fits

        h = fits.open(map_path)
        try:
            obsids = [float(o.split("_")[0]) for o in h[0].header["OBSIDS"].split(",")]
            self.maps_included = obsids
        except Exception as e:
            print(e, "Failed to read OBSIDS")

        try:
            self.coadd_days = float(h[0].header["NDAYS"])
        except Exception as e:
            print(e, "Failed to read NDAYS")

        try:
            self.freq = h[0].header["FREQ"]
        except Exception as e:
            print(e, "Failed to read FREQ")
            self.freq = str(map_path).split("/")[-1].split("_")[3]

        if not self.res:
            self.res = np.abs(self.rho_map.wcs.wcs.cdelt[0]) * degree
        del h

        return

    def coadd_maps(self, second_map, box: np.ndarray = None):
        ## input map should be /file/path/to/obsid_arr_freq_[rho/kappa].fits
        ## only supports I map right now
        ## must be Path object
        from os.path import exists

        if not isinstance(second_map, Path):
            raise TypeError("second_map must be a Path object.")
        if "rho" in str(second_map):
            maptype = "rho"
        elif "kappa" in str(second_map):
            maptype = "kappa"
        ## could do something with intensity and inverse variance maps
        ## or even just intensity map + calculated inverse variance...
        elif "map" in str(second_map):
            maptype = "map"
        elif "ivar" in str(second_map):
            maptype = "ivar"

        if isinstance(self.rho_map, type(None)):
            if not self.res:
                self.res = (
                    np.abs(enmap.read_map_geometry(str(second_map))[1].wcs.cdelt[0])
                    * degree
                )
            self.init_empty_so_map(res=self.res, box=box)
            self.get_map_info(second_map)

        path2map = str(second_map).split(f"{maptype}.fits")[0]
        if (
            not exists(path2map + "rho.fits")
            or not exists(path2map + "kappa.fits")
            or not exists(path2map + "time.fits")
        ):
            raise FileNotFoundError(
                f"One of {path2map} [rho/kappa/time].fits not found! Cant coadd."
            )

        arr2 = path2map.split("/")[-1].split("_")[2]
        freq2 = path2map.split("/")[-1].split("_")[3]
        ctime2 = float(path2map.split("/")[-1].split("_")[1])

        second_rho = enmap.read_map(
            path2map + "rho.fits", sel=0, box=box
        )  # whatever rho is, only I
        self.load_map(self.rho_map, box=box)
        self.rho_map.insert(second_rho, iwcs=second_rho.wcs, op=lambda a, b: a + b)
        del second_rho

        second_kappa = enmap.read_map(
            path2map + "kappa.fits", sel=0, box=box
        )  # whatever kappa is, only I
        self.load_map(self.kappa_map, box=box)
        self.kappa_map.insert(
            second_kappa, iwcs=second_kappa.wcs, op=lambda a, b: a + b
        )

        second_time = enmap.read_map(path2map + "time.fits", box=box)  # time map
        t0 = get_observation_start_time(second_map)
        if not t0:
            t0 = 0.0
        self.time_map.insert(
            (second_time + t0) * second_kappa,
            iwcs=second_time.wcs,
            op=lambda a, b: a + b,
        )
        del second_time, second_kappa

        if isinstance(self.maps_included, bool):
            mapinfo = f"{self.map_ctime}_{self.wafer_name}_{self.freq}"
            self.maps_included = [mapinfo]
        self.maps_included.append(f"{ctime2}_{arr2}_{freq2}")

        return

    def subtract_sources(
        self,
        sources: list,
        src_model: enmap.ndmap = None,
        verbose=False,
        cuts={},
    ):
        """
        src_model is a simulated (model) map of the sources in the list.
        sources are fit using photutils, and are SourceCandidate objects
        with fwhm_a, fwhm_b, ra, dec, flux, and orientation
        """
        if len(sources) == 0:
            return
        if not isinstance(self.flux, enmap.ndmap):
            raise ValueError("self.flux is None, cannot subtract sources.")
        if not isinstance(src_model, enmap.ndmap):
            from ..utils.utils import get_fwhm

            src_model = make_model_source_map(
                self.flux,
                sources,
                nominal_fwhm_arcmin=get_fwhm(self.freq),
                verbose=verbose,
                cuts=cuts,
            )

        self.flux -= src_model
        ## mask the snr map as well since we've subtracted the sources we want to ignore them.
        self.snr[abs(src_model) > 1e-8] = 0.0
        return src_model


def enmap_map_union(map1, map2):
    ## from enmap since enmap.zeros causes it to fail.
    oshape, owcs = enmap.union_geometry([map1.geometry, map2.geometry])
    omap = enmap.zeros(map1.shape[:-2] + oshape[-2:], owcs, map1.dtype)
    omap.insert(map1)
    omap.insert(map2, op=lambda a, b: a + b)
    return omap


def load_map(
    map_path: Optional[Path | str | list] = None,
    map_sim_params: Optional[dict] = None,
    box: Optional[np.ndarray] = None,
    use_map_geometry: bool = False,
    ctime: float = 0.0,
    verbose: bool = False,
) -> Optional[Depth1Map]:
    """
    Unified function to load a map from file or generate a simulation.

    Args:
        map_path: Path to the map file (.fits)
        map_sim_params: Parameters for simulated map (if map_path is None)
        box: Bounding box in the form [[ra_min,dec_min],[ra_max,dec_max]]
        use_map_geometry: Whether to use the geometry of the map
        ctime: Creation time for simulated map
        verbose: Whether to print verbose messages

    Returns:
        Depth1Map object or None if loading fails
    """
    # Check if we're loading a simulated map
    # if using map geometry we'll want to load the actual map.
    if map_sim_params and (isinstance(map_path, type(None)) or not use_map_geometry):
        from ..sims import sim_maps as mapsims

        try:
            empty_map = mapsims.make_enmap(
                center_ra=map_sim_params["maps"]["center_ra"],
                center_dec=map_sim_params["maps"]["center_dec"],
                width_ra=map_sim_params["maps"]["width_ra"],
                width_dec=map_sim_params["maps"]["width_dec"],
            )
            sim_map = Depth1Map(
                wafer_name=map_sim_params["array_info"]["arr"],
                freq=map_sim_params["array_info"]["freq"],
                map_ctime=ctime,
                map_start_time=0.0,
                res=np.abs(empty_map.wcs.wcs.cdelt[0]) * degree,
            )
            sim_map.flux = empty_map + mapsims.make_noise_map(
                empty_map, map_sim_params["maps"]["map_noise"]
            )
            sim_map.snr = empty_map + np.ones(empty_map.shape)
            sim_map.time_map = empty_map + np.ones(empty_map.shape) * ctime
            return sim_map
        except Exception as e:
            if verbose:
                print(f"Failed to create simulated map: {e}")
            return None

    # No map path provided
    if isinstance(map_path, type(None)):
        if verbose:
            print("No map path or simulation parameters provided")
        return None

    if isinstance(map_path, (Path, str)):
        # Check if the map path exists
        from os.path import exists

        if not exists(map_path):
            if verbose:
                print(f"Map path does not exist: {str(map_path)}")
            return None

    # Load map based on format
    try:
        if "map.fits" in str(map_path):
            # Load intensity map format
            try:
                imap = enmap.read_map(str(map_path), sel=0, box=box)  # intensity map
            except Exception:
                imap = enmap.read_map(str(map_path), box=box)

            # Check if map is all zeros or NaN
            if np.all(imap == 0.0) or np.all(np.isnan(imap)):
                if verbose:
                    print("Map is all nan or zeros, skipping")
                return None

            path = str(map_path).split("map.fits")[0]

            # Check if other required files exist
            if not exists(path + "ivar.fits") or not exists(path + "time.fits"):
                if verbose:
                    print(
                        f"Missing required files: ivar.fits or time.fits for {map_path}"
                    )
                return None

            ivar = enmap.read_map(path + "ivar.fits", box=box)  # inverse variance map
            time = enmap.read_map(path + "time.fits", box=box)  # time map

            # Extract metadata
            t0 = get_observation_start_time(map_path)
            arr = path.split("/")[-1].split("_")[2]
            freq = path.split("/")[-1].split("_")[3]
            ctime = float(path.split("/")[-1].split("_")[1])
            res = np.abs(imap.wcs.wcs.cdelt[0]) * degree

            return Depth1Map(
                intensity_map=imap,
                inverse_variance_map=ivar,
                time_map=time + t0,
                wafer_name=arr,
                freq=freq,
                map_ctime=ctime,
                res=res,
                map_start_time=t0,
                box=box,
            )

        elif "rho.fits" in str(map_path):
            # Load rho map format
            try:
                rho = enmap.read_map(str(map_path), sel=0, box=box)  # rho map
            except Exception:
                rho = enmap.read_map(str(map_path), box=box)

            # Check if map is all zeros or NaN
            if np.all(rho == 0.0) or np.all(np.isnan(rho)):
                if verbose:
                    print("Rho map is all nan or zeros, skipping")
                return None

            path = str(map_path).split("rho.fits")[0]

            # Check if other required files exist
            if not exists(path + "kappa.fits") or not exists(path + "time.fits"):
                if verbose:
                    print(
                        f"Missing required files: kappa.fits or time.fits for {map_path}"
                    )
                return None

            try:
                kappa = enmap.read_map(path + "kappa.fits", sel=0, box=box)
            except Exception:
                kappa = enmap.read_map(path + "kappa.fits", box=box)

            try:
                time = enmap.read_map(path + "time.fits", box=box)
            except Exception:
                # Time map shouldn't have different polarizations, but just in case
                time = enmap.read_map(path + "time.fits", sel=0, box=box)

            # Extract metadata
            arr = path.split("/")[-1].split("_")[2]
            freq = path.split("/")[-1].split("_")[3]
            ctime = float(path.split("/")[-1].split("_")[1])
            res = np.abs(rho.wcs.wcs.cdelt[0]) * degree
            t0 = get_observation_start_time(map_path)

            return Depth1Map(
                rho_map=rho,
                kappa_map=kappa,
                time_map=time + t0,
                wafer_name=arr,
                freq=freq,
                map_ctime=ctime,
                res=res,
                map_start_time=t0,
                box=box,
            )

        else:
            if verbose:
                print(f"Unsupported map format: {map_path}")
            return None

    except Exception as e:
        if verbose:
            print(f"Error loading map {map_path}: {e}")
        return None


def kappa_clean(kappa: np.ndarray, rho: np.ndarray, min_frac: float = 1e-3):
    kappa = np.maximum(kappa, np.nanmax(kappa) * min_frac)
    kappa[np.where(rho == 0.0)] = 0.0
    return kappa


def clean_map(
    imap: np.ndarray,
    inverse_variance: np.ndarray,
    fraction: float = 0.01,
    cut_on: str = "max",
):
    ## cut_on can be max or median, this sets the imap to zero for values of inverse variance
    ## which are below fraction*max or fraction*median of inverse variance map.
    if cut_on == "median" or cut_on == "med":
        imap[inverse_variance < (np.nanmedian(inverse_variance) * fraction)] = 0
    elif cut_on == "percentile" or cut_on == "pct":
        imap[inverse_variance < np.nanpercentile(inverse_variance, fraction)] = 0
    else:
        if cut_on != "max":
            print("%s cut_on not supported, defaulting to max cut" % cut_on)
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


def get_maptype(map_path: Path):
    if "rho" in str(map_path):
        maptype = "rho"
    elif "kappa" in str(map_path):
        maptype = "kappa"
    elif "map" in str(map_path):
        maptype = "map"
    elif "ivar" in str(map_path):
        maptype = "ivar"
    elif "time" in str(map_path):
        maptype = "time"
    if isinstance(map_path, str):
        suffix = "." + map_path.split(".")[-1]
    else:
        suffix = map_path.suffix
    return maptype, suffix


def get_observation_start_time(map_path: Path):
    from os.path import exists

    maptype, _ = get_maptype(map_path)
    infofile = str(map_path).split(maptype)[0] + "info.hdf"
    if not exists(infofile):
        t0 = 0.0
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


def get_submap(
    imap: enmap.ndmap,
    ra_deg: float,
    dec_deg: float,
    size_deg: float = 0.5,
) -> enmap:
    ## Does not reproject
    from pixell.utils import degree

    ra = ra_deg * degree
    dec = dec_deg * degree
    radius = size_deg * degree
    omap = imap.submap(
        [[dec - radius, ra - radius], [dec + radius, ra + radius]],
    )
    return omap


def get_thumbnail(
    imap: enmap.ndmap,
    ra_deg: float,
    dec_deg: float,
    size_deg: float = 0.5,
    proj: str = "tan",
) -> enmap:
    from pixell import reproject
    from pixell.utils import degree

    ra = ra_deg * degree
    dec = dec_deg * degree
    omap = reproject.thumbnails(
        imap,
        [dec, ra],
        size_deg * degree,
        proj=proj,
    )
    if np.all(np.isnan(omap)):
        omap = reproject.thumbnails(
            np.nan_to_num(imap),
            [dec, ra],
            size_deg * degree,
            proj=proj,
        )
    return omap


def get_time_safe(time_map: enmap.ndmap, poss: list, r: float = 5.0):
    ## pos [[dec,ra],[dec,ra],...]
    ## r radius in arcmin
    from pixell.utils import arcmin

    r *= arcmin
    poss = np.array(poss)
    vals = time_map.at(poss, order=0)
    bad = np.where(vals == 0)[0]
    if len(bad) > 0:
        pixboxes = enmap.neighborhood_pixboxes(
            time_map.shape, time_map.wcs, poss.T[bad], r=r
        )
        for i, pixbox in enumerate(pixboxes):
            thumb = time_map.extract_pixbox(pixbox)
            mask = thumb != 0
            vals[bad[i]] = np.sum(mask * thumb) / np.sum(mask)
    return vals


def widefield_geometry(
    res=None,
    shape=None,
    dims=(),
    proj="car",
    variant="fejer1",
    include_half_pixel_offset=False,
):
    """Build an enmap covering the full sky, with the outermost pixel centers
        at the poles and wrap-around points. Only the car projection is
        supported for now, but the variants CC and fejer1 can be selected using
        the variant keyword. This currently defaults to CC, but will likely
    change to fejer1 in the future.
    """
    from pixell import utils as pixell_utils
    from pixell import wcsutils

    if variant.lower() == "cc":
        yo = 1
    elif variant.lower() == "fejer1":
        yo = 0
    else:
        raise ValueError("Unrecognized CAR variant '%s'" % str(variant))

    # Set up the shape/resolution
    ra_width = 2 * np.pi
    dec_width = 80 * np.pi / 180.0  ## -60 to +20
    ra_cent = 0
    dec_cent = -20
    if shape is None:
        res = np.zeros(2) + res
        shape = pixell_utils.nint(([dec_width, ra_width] / res) + (yo, 0))
    else:
        res = np.array([dec_width, ra_width]) / (np.array(shape) - (yo, 0))
    ny, nx = shape
    assert abs(res[0] * (ny - yo) - dec_width) < 1e-8, (
        "Vertical resolution does not evenly divide the sky; this is required for SHTs."
    )
    assert abs(res[1] * nx - ra_width) < 1e-8, (
        "Horizontal resolution does not evenly divide the sky; this is required for SHTs."
    )
    wcs = wcsutils.WCS(naxis=2)
    # Note the reference point is shifted by half a pixel to keep
    # the grid in bounds, from ra=180+cdelt/2 to ra=-180+cdelt/2.
    #
    wcs.wcs.crval = [0, 0]
    if include_half_pixel_offset:
        wcs.wcs.crval = [res[1] / 2 / pixell_utils.degree, 0]
    wcs.wcs.cdelt = [
        -ra_width / pixell_utils.degree / nx,
        dec_width / pixell_utils.degree / (ny - yo),
    ]
    wcs.wcs.crpix = [nx // 2, (ny - dec_cent * pixell_utils.degree / res[0]) / 2]  #
    if include_half_pixel_offset:
        wcs.wcs.crpix = [nx // 2 + 0.5, (ny + 1) / 2]  #
    wcs.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    return dims + (ny, nx), wcs


def flat_field_using_pixell(
    mapdata,
    tilegrid=1.0,
):
    """
    Use the tiles module to do map tiling using scan strategy.
    Not sure how much better (if at all) this is than using background2d
    """
    from .tiles import get_medrat, get_tmap_tiles

    try:
        med_ratio = get_medrat(
            mapdata.snr,
            get_tmap_tiles(
                np.nan_to_num(mapdata.time_map) - np.nanmin(mapdata.time_map),
                tilegrid,
                mapdata.snr,
                id=f"{mapdata.freq}",
            ),
        )

        # mapdata.flux*=med_ratio
        mapdata.snr *= med_ratio
        mapdata.flatfielded = True
    except Exception as e:
        print(
            e,
            " Cannot flatfield at this time... need to update medrat algorithm for coadds",
        )

    return


def flat_field_using_photutils(
    mapdata: Depth1Map,
    tilegrid: float = 1.0,
    mask: enmap.ndmap = None,
    sigmaclip: float = 5.0,
):
    """
    use photutils.background.Background2D to calculate the rms in tiles.
    get the rms of the tiled snr map (i.e. the rms should be 1)
    calculate the median.
    divide the snr map by the tiled rms / median.

    tilegrid is in degrees, and is the size of the tile to use for the background
    """
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, StdBackgroundRMS
    from pixell.utils import degree

    sigmaclip = SigmaClip(sigma=sigmaclip) if sigmaclip else None
    try:
        background = Background2D(
            mapdata.snr,
            int(tilegrid * degree / mapdata.res),
            sigma_clip=sigmaclip,
            bkg_estimator=StdBackgroundRMS(sigmaclip),
            mask=mask,
        )
        relative_rms = background.background_rms / background.background_rms_median
        mapdata.snr /= relative_rms
        mapdata.flatfielded = True
    except Exception as e:
        print(e, " Failed to flatfield using photutils")
        mapdata.flatfielded = False

    return


def preprocess_map(
    mapdata,
    galmask_file="/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/mask_for_sources2019_plus_dust.fits",
    tilegrid=0.5,
    flatfield_method="photutils",
    output_units="Jy",
    edge_mask_arcmin=10,
    sigmaclip=5.0,
    skip=False,
    log=None,
):
    """
    Preprocess the map by cleaning, flatfielding, and masking.
    Converts the rho, kappa maps to flux and snr maps.

    Inputs:
    - mapdata: Depth1Map object
    - galmask_file: path to the galaxy mask file
    - tilegrid: size of the tile to use for flatfielding
    - flatfield_method: method to use for flatfielding ('photutils' or 'pixell')
    - output_units: units of the output map ('Jy' or 'mJy')
    - edge_mask_arcmin: size of the edge mask in arcminutes
    - sigmaclip: sigma clip value for flatfielding
    - skip: skip this function, useful if simulated map


    """
    if skip:
        return
    ## tilegrid in deg, used for setting median ratio flatfielding grid size
    from pixell.enmap import read_map

    from .masks import mask_dustgal, mask_edge

    log = log.new()
    log.info("preprocess.start")
    if not mapdata.cleaned:
        ## ignore divide by zero warning since that will happen outside the weighted region
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mapdata.kappa_map = kappa_clean(mapdata.kappa_map, mapdata.rho_map)
            mapdata.rho_map = clean_map(
                mapdata.rho_map, mapdata.kappa_map, cut_on="median", fraction=0.05
            )
        mapdata.cleaned = True
        log.info("preprocess.cleaned")

    ## ignore divide by zero warning since that will happen outside the weighted region
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mapdata.flux = mapdata.rho_map / mapdata.kappa_map
        mapdata.snr = mapdata.rho_map * mapdata.kappa_map ** (-0.5)
    mapdata.kappa_map = None
    mapdata.rho_map = None
    log.info("preprocess.flux_snr_calculated")
    if (
        np.all(np.isnan(mapdata.flux))
        or np.all(np.isnan(mapdata.snr))
        or np.all(mapdata.snr == 0.0)
        or np.all(mapdata.flux == 0.0)
    ):
        log.warning("preprocess.flux_snr_all_nan_or_zero")
        mapdata = None
        return

    if not mapdata.masked:
        mapdata.masked = 0
        mask = enmap.ones(mapdata.flux.shape, wcs=mapdata.flux.wcs)

        log.info("preprocess.masking", galaxy_mask=galmask_file)
        try:
            galaxy_mask = mask_dustgal(
                mapdata.flux,
                read_map(
                    galmask_file,
                    box=mapdata.box,
                ),
            )
            mapdata.flux *= galaxy_mask
            mapdata.snr *= galaxy_mask
            mask *= galaxy_mask
            del galaxy_mask
            log.info("preprocess.masking.galaxy")
        except Exception as e:
            log.warning("preprocess.masking.galaxy.failed", error=e)
            mapdata.masked += 1

        try:
            ## could add in planet mask here:
            planet_mask = None
            if planet_mask:
                mapdata.flux *= planet_mask
                mapdata.snr *= planet_mask
                mask *= planet_mask
            del planet_mask

        except Exception as e:
            log.warning("preprocess.masking.planet.failed", error=e)
            mapdata.masked += 1

        try:
            edge_mask = mask_edge(
                mapdata.flux, edge_mask_arcmin / (mapdata.res / arcmin)
            )
            mapdata.flux *= edge_mask
            mapdata.snr *= edge_mask
            mask *= edge_mask
            del edge_mask
            log.info("preprocess.masking.edge", edge_mask_arcmin=edge_mask_arcmin)
        except Exception as e:
            log.warning("preprocess.masking.edge.failed", error=e)
            mapdata.masked += 1

    if (
        np.all(np.isnan(mapdata.flux))
        or np.all(np.isnan(mapdata.snr))
        or np.all(mapdata.snr == 0.0)
        or np.all(mapdata.flux == 0.0)
    ):
        log.warning("preprocess.flux_snr_all_nan_or_zero_after_masking")
        mapdata = None
        return

    if not mapdata.flatfielded:
        log.info(
            "preprocess.flatfielding",
            method=flatfield_method,
            tilegrid=tilegrid,
        )
        if flatfield_method == "photutils":
            flat_field_using_photutils(
                mapdata,
                tilegrid=tilegrid,
                mask=1.0 - mask,
                sigmaclip=sigmaclip,
            )
        elif flatfield_method == "pixell":
            flat_field_using_pixell(
                mapdata,
                tilegrid=tilegrid,
            )
        else:
            log.error("preprocess.flatfielding.failed", method=flatfield_method)
            raise ValueError("Flatfield method %s not supported" % flatfield_method)

        del mask
        log.info("preprocess.flatfielding.success")
    if output_units == "Jy":
        mapdata.flux /= 1000.0
    log.bind(map_flux_units=output_units)
    log.info("preprocess.complete")
    return


def make_model_source_map(
    imap: enmap.ndmap,
    sources: list,
    nominal_fwhm_arcmin: float = None,
    matched_filtered=False,
    verbose=False,
    cuts={},
):
    """
    Use source list containing fwhm_a, fwhm_b, ra, dec, flux, and orientation
    to create a model map of the sources.
    This is used to subtract the sources from the map.

    Arguments:
        - imap: enmap object
        - sources: list of source candidates
        - nominal_fwhm_arcmin: nominal fwhm in arcmin
        - matched_filtered: if True, then the fwhm is matched filtered
        - verbose: if True, then print out the sources that are not included
        - cuts: dictionary of cuts to apply to the sources
    Returns:
        - model_map: enmap object of the model map
    """
    if len(sources) == 0:
        return imap
    from photutils.datasets import make_model_image
    from photutils.psf import GaussianPSF
    from pixell.utils import arcmin, degree

    from ..sims.sim_utils import make_2d_gaussian_model_param_table

    if matched_filtered:
        nominal_fwhm_arcmin *= np.sqrt(2)

    res_arcmin = abs(imap.wcs.wcs.cdelt[0] * degree / arcmin)

    model_params = make_2d_gaussian_model_param_table(
        imap,
        sources,
        nominal_fwhm_arcmin=nominal_fwhm_arcmin,
        cuts=cuts,
        verbose=verbose,
    )

    shape = (
        int(5 * nominal_fwhm_arcmin / res_arcmin),
        int(5 * nominal_fwhm_arcmin / res_arcmin),
    )
    model_map = make_model_image(
        imap.shape,
        GaussianPSF(),
        model_params,
        model_shape=shape,
    )

    return model_map
