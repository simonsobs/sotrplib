from pixell.enmap import enmap


class SourceCatalog:
    def __init__(self, ra, dec):
        self.ra = ra
        self.dec = dec


def load_act_catalog(
    source_cat_file: str = "/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/PS_S19_f090_2pass_optimalCatalog.fits",
    flux_threshold: float = 0,
    log=None,
):
    """
    source_cat_file is path to source catalog

    flux_threshold is a threshold, in Jy, below which we ignore the sources.
        i.e. if we want sources in depth-1 map, we don't care about sub mJy sources.
        by default, returns all positive sources.
    """
    from astropy.table import Table

    log.bind(func_name="load_act_catalog")
    sourcecat = None
    sourcecat = Table.read(source_cat_file)
    sources = sourcecat[sourcecat["fluxJy"] >= (flux_threshold)]
    sources["RADeg"][sources["RADeg"] < 0] += 360.0
    log.info(
        "load_catalog.act_catalog.loaded",
        total_sources=len(sources["decDeg"]),
        flux_thresh_mJy=flux_threshold * 1000,
    )
    out_dict = {key: sources[key] for key in sources.colnames}

    return out_dict


def convert_gauss_fit_to_source_cat(
    gauss_fits: list, uncert_prefix: str = "err_", log=None
):
    """
    gauss fits is a list of dictionaries of the output params of the gaussian fitting.
    convert that into a dictionary of lists.

    since there are uncertainties on the fits, make keys err_[blah] for those fits.
    this is inspired by fluxJy and err_fluxJy in act table.
    """
    log.bind(func_name="convert_gauss_fit_to_source_cat")

    if not isinstance(gauss_fits, list):
        log.error("convert_gauss_fit_to_source_cat.not_list", gauss_fits=gauss_fits)
        raise ValueError("gauss_fits should be a list of dictionaries")
    if not gauss_fits:
        log.error("convert_gauss_fit_to_source_cat.empty", gauss_fits=gauss_fits)
        return {}

    log.info("convert_gauss_fit_to_source_cat.start", gauss_fits_length=len(gauss_fits))
    sources = {}
    for i in range(len(gauss_fits)):
        for key in gauss_fits[i]:
            if key not in sources:
                sources[key] = []
                sources[uncert_prefix + key] = []
            if isinstance(gauss_fits[i][key], tuple):
                keyval, keyvaluncert = gauss_fits[i][key]
                sources[key].append(keyval)
                sources[uncert_prefix + key].append(keyvaluncert)
            else:
                keyval = gauss_fits[i][key]
                sources[key].append(keyval)
    popkeys = [k for k in sources if not sources[k]]
    for k in popkeys:
        sources.pop(k)
    if "name" not in sources:
        sources["name"] = sources["sourceID"]
    log.info("convert_gauss_fit_to_source_cat.success", sources=len(sources))
    return sources


def convert_json_to_act_format(json_list):
    """
    json list is the list of dictionaries output to json file format.
    this will hopefully be depricated when using the database.

    there are other differences too, but these are the ones we care about now.
    """
    import numpy as np

    sources = {}
    for i in range(len(json_list)):
        for key in json_list[i]:
            if key not in sources:
                sources[key] = [json_list[i][key]]
            else:
                sources[key].append(json_list[i][key])
    for key in sources:
        sources[key] = np.asarray(sources[key])

    sources["RADeg"] = sources["ra"]
    del sources["ra"]

    sources["decDeg"] = sources["dec"]
    del sources["dec"]

    sources["fluxJy"] = sources["flux"] / 1000.0
    del sources["flux"]

    sources["err_fluxJy"] = sources["dflux"] / 1000.0
    del sources["dflux"]

    sources["name"] = sources["crossmatch_name"]
    del sources["crossmatch_name"]

    return sources


def load_json_test_catalog(source_cat_file: str, flux_threshold: float = 0, log=None):
    """
    Load the output file with SourceCandidate dictionaries
    which is stored as a json file.

    flux_threshold is a threshold, in Jy, below which we ignore the sources.
        i.e. if we want sources in depth-1 map, we don't care about sub mJy sources.
        by default, returns all positive sources.
    """
    import json

    log = log.bind(func_name="load_json_test_catalog")

    with open(source_cat_file, "r") as f:
        # Load the JSON data into a Python dictionary
        data = [json.loads(line) for line in f]
    log.info(
        "load_json_test_catalog.file_loaded",
        source_cat_file=source_cat_file,
        data_length=len(data),
        flux_threshold_Jy=flux_threshold,
    )
    sources = convert_json_to_act_format(data)
    sources["RADeg"][sources["RADeg"] < 0] += 360.0

    flux_cut = sources["fluxJy"] >= flux_threshold
    for key in sources:
        sources[key] = sources[key][flux_cut]
    log.info("load_json_test_catalog.data_converted")
    return sources


def load_websky_csv_catalog(source_cat_file: str, flux_threshold: float = 0, log=None):
    """
    load the websky catalog from a csv containing the columns
    flux(Jy), ra(deg), dec(deg)
    """
    from numpy import asarray, loadtxt

    from ..utils.utils import radec_to_str_name

    log.bind(func_name="load_websky_csv_catalog")
    websky_flux, websky_ra, websky_dec = loadtxt(
        source_cat_file, delimiter=",", unpack=True, skiprows=1
    )
    websky_ra[websky_ra > 180.0] -= 360
    inds = websky_flux > flux_threshold
    log.info(
        "load_websky_csv_catalog.sources_above_threshold",
        data_length=sum(inds),
        flux_threshold=flux_threshold,
    )
    sources = {}
    sources["RADeg"] = websky_ra[inds]
    sources["decDeg"] = websky_dec[inds]
    sources["fluxJy"] = websky_flux[inds]
    sources["err_fluxJy"] = websky_flux[inds] * 0.0
    sources["name"] = asarray(
        [
            radec_to_str_name(sources["RADeg"][i], sources["decDeg"][i])
            for i in range(sum(inds))
        ]
    )
    log.info(
        "load_websky_csv_catalog.catalog_loaded", total_sources=len(sources["name"])
    )
    return sources


def load_pandas_catalog(
    source_cat_file: str,
    flux_threshold: float = 0,
    log=None,
):
    """
    load the source catalog from a pandas dataframe stored in a pickle file.
    """
    import pandas as pd

    log.bind(func_name="load_pandas_catalog")
    sources = pd.read_pickle(source_cat_file)
    sources["RADeg"][sources["RADeg"] < 0] += 360.0
    flux_cut = sources["fluxJy"] >= flux_threshold
    sources = sources[flux_cut]
    sources["name"] = sources["sourceID"]
    log.info(
        "load_pandas_catalog.data_converted",
        n_sources=len(sources["name"]),
        flux_threshold=flux_threshold,
    )

    return sources


def load_million_quasar_catalog(
    source_cat_file: str = "/scratch/gpfs/SIMONSOBS/users/amfoster/scratch/milliquas.fits",
    log=None,
):
    from astropy.table import Table

    log.bind(func_name="load_million_quasar_catalog")
    sources = Table.read(source_cat_file)
    sources.rename_column("RA", "RADeg")
    sources.rename_column("DEC", "decDeg")
    log.info("load_million_quasar_catalog.catalog_loaded", total_sources=len(sources))
    return sources


def load_catalog(
    source_cat_file: str,
    flux_threshold: float = 0,
    mask_outside_map: bool = False,
    mask_map: enmap = None,
    return_source_cand_list: bool = False,
    log=None,
):
    """
    flux_threshold is a threshold, in Jy, below which we ignore the sources.
        i.e. if we want sources in depth-1 map, we don't care about sub mJy sources.
        by default, returns all positive sources.
    mask_outside_map: bool
        if False, do not mask the mapped region, just include all sources in catalog.
    mask_map: enmap
        map with which to do the masking; can be anything that is zero/nan outside observed region.

    Returns:

        sources: source catalog, in astropy.table.table.Table or dict format... dumb but works for now.

    """

    ##
    ##need a way to load frequency, array, ctime information.
    ##
    log = log.bind(func_name="load_catalog")
    if ".pkl" in source_cat_file:
        sources = load_pandas_catalog(
            source_cat_file=source_cat_file,
            flux_threshold=flux_threshold,
            log=log,
        )
    if (
        "PS_S19_f090_2pass_optimalCatalog.fits" in source_cat_file
        or "catmaker" in source_cat_file
    ):
        sources = load_act_catalog(
            source_cat_file=source_cat_file, flux_threshold=flux_threshold, log=log
        )
    if "websky_cat_100_1mJy.csv" in source_cat_file:
        sources = load_websky_csv_catalog(
            source_cat_file=source_cat_file, flux_threshold=flux_threshold, log=log
        )
    if ".json" in source_cat_file:
        sources = load_json_test_catalog(
            source_cat_file=source_cat_file, flux_threshold=flux_threshold, log=log
        )
    log.info(
        "load_catalog.loaded",
        source_cat_file=source_cat_file,
        total_sources=len(sources),
    )

    if mask_outside_map and not isinstance(mask_map, type(None)):
        from ..sources.finding import mask_sources_outside_map

        source_mask = mask_sources_outside_map(sources, mask_map)
        if isinstance(sources, dict):
            for key in sources:
                sources[key] = sources[key][source_mask]
        else:
            sources = sources[source_mask]
        log.info("load_catalog.load_in_map", total_in_map_sources=sum(source_mask))
    if return_source_cand_list:
        from ..sources.forced_photometry import convert_catalog_to_source_objects

        sources = convert_catalog_to_source_objects(sources, log=log)

    log.info("load_catalog.complete")
    return sources


def write_json_catalog(
    outcat, out_dir: str = "./", out_name: str = "source_catalog.json"
):
    with open(out_dir + out_name, "w") as f:
        for oc in outcat:
            json_string_cand = oc.json()
            f.write(json_string_cand)
            f.write("\n")
    return
