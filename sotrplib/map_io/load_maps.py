from pathlib import Path

import numpy as np
from os import path
from ..maps.maps import Depth1Map


def get_depth1_mapset(map_path: Path | str) -> dict[str, Path]:
    """
    Converts the base map path (which is to the `/file/.../..._map.fits`
    main map file) to a dictionary of all maps based upon the unified
    naming scheme.
    """
    if not isinstance(map_path, Path):
        map_path = Path(map_path)

    directory = map_path.parent
    filename = map_path.name

    return {
        maptype: directory / filename.replace("_map.fits", f"_{maptype}.fits")
        for maptype in ["map", "ivar", "rho", "time", "kappa"]
    }


def load_depth1_mapset(map_path: Path | str,
                        ivar_path: Path | str | None = None,
                        rho_path: Path | str | None = None,
                        kappa_path: Path | str | None = None,
                        time_path: Path | str | None = None,
                        polarization_selector: int = None,
                    ) -> Depth1Map | None:
    """

    Load a set of depth1 maps. This can be :
     1) map.fits and ivar.fits (i.e. the inverse-variance weighted unfiltered map, and the inverse-variance map)
     2) rho.fits and kappa.fits (i.e. the rho and kappa maps)

    both options should also have a time.fits map containing the time of observation.

    Parameters
    ----------
    map_path: Path | str
        The map to the intensity map.
    ivar_path: Path | str | None
        The path to the inverse variance map. If not present, we assume
        that the ivar_path is the same as map_path, just swapping `map.fits`
        for `ivar.fits`.
    rho_path: Path | str | None
        The path to the rho map. If not present, we assume
        that the rho_path is the same as map_path, just swapping `map.fits`
        for `rho.fits`.
    kappa_path: Path | str | None
        The path to the kappa map. If not present, we assume
        that the kappa_path is the same as map_path, just swapping `map.fits`
        for `kappa.fits`.
    time_path: Path | str | None
        The path to the time map. If not present, we assume
        that the time_path is the same as map_path, just swapping `map.fits`
        for `time.fits`.
    polarization_selector: int = 0
        Polarization selector: 0 for I, 1 for Q, 2 for U. Default is 0 (I).

    Returns
    -------
    Depth1Map | None
        A Depth1Map object containing the maps, or None if the map is all zeros.
    """

    map_paths = get_depth1_mapset(map_path=map_path)
    if ivar_path is not None:
         map_paths["ivar"] = ivar_path
    if rho_path is not None:
         map_paths["rho"] = rho_path
    if kappa_path is not None:
         map_paths["kappa"] = kappa_path
    if time_path is not None:
         map_paths["time"] = time_path

    def _read_map(name, value, sel=None):
        from pixell.enmap import read_map
        if value is not None:
            map_paths[name] = Path(value)
        # pixell only supports string filenames
        return read_map(str(map_paths[name]), sel=sel)

    ## check if imap exists, otherwise read in rho, kappa
    if not map_paths["map"].exists():
        imap=None
        ivar=None
    else:
        imap = _read_map("map", map_paths["map"], sel=polarization_selector)
        ivar = _read_map("ivar", map_paths["ivar"], sel=polarization_selector)
        # check if map is all zeros
        if np.all(imap == 0.0) or np.all(np.isnan(imap)):
            print("map is all nan or zeros, skipping")
            return None

    if not path.exists(map_paths['rho']):
        rho_path = None
        kappa_path = None
    else:
        rho = _read_map("rho", map_paths["rho"], sel=polarization_selector)
        kappa = _read_map("kappa", map_paths["kappa"], sel=polarization_selector)
    
    map_filename = map_paths["map"].name
    if not path.exists(map_paths["time"]):
        time = None
    else:
        time = _read_map("time", map_paths["time"])

    ## These should be contained in the map metadata in the future
    map_info = map_filename.split("_")
    if len(map_info) < 3:
        arr = None
        freq = None
        ctime= None
    else:
        arr = map_info[2]
        freq = map_info[3]
        ctime = float(map_info[1])

    return Depth1Map(
        intensity_map=imap,
        inverse_variance_map=ivar,
        rho_map=rho,
        kappa_map=kappa,
        time_map=time,
        wafer_name=arr,
        freq=freq,
        map_ctime=ctime,
    )
