from pathlib import Path

import numpy as np
from pixell.enmap import read_map

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
        x: directory / filename.replace("_map.fits", f"_{x}.fits")
        for x in ["map", "ivar", "rho", "time", "kappa"]
    }


def load_depth1_mapset(
    map_path: Path | str,
    ivar_path: Path | str | None = None,
    rho_path: Path | str | None = None,
    kappa_path: Path | str | None = None,
    time_path: Path | str | None = None,
    polarization_selector: int = 0,
) -> Depth1Map | None:
    """
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

    def _read_map(name, value, sel=None):
        if value is not None:
            map_paths[name] = Path(value)

        # pixell only supports string filenames
        return read_map(str(map_paths[name]), sel=sel)

    imap = _read_map("map", map_path, sel=polarization_selector)

    # check if map is all zeros
    if np.all(imap == 0.0) or np.all(np.isnan(imap)):
        print("map is all nan or zeros, skipping")
        return None

    ivar = _read_map("ivar", ivar_path)
    rho = _read_map("rho", rho_path, sel=polarization_selector)
    kappa = _read_map("kappa", kappa_path, sel=polarization_selector)
    time = _read_map("time", time_path)

    # check if map is all zeros
    if np.all(imap == 0.0) or np.all(np.isnan(imap)):
        print("map is all nan or zeros, skipping")
        return None

    map_filename = map_paths["map"].name

    ## These should be contained in the map metadata in the future
    arr = map_filename.split("_")[2]
    freq = map_filename.split("_")[3]
    ctime = float(map_filename.split("_")[1])

    return Depth1Map(
        imap=imap,
        ivar=ivar,
        rho=rho,
        kappa=kappa,
        time=time,
        arr=arr,
        freq=freq,
        ctime=ctime,
    )
