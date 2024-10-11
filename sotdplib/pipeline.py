## DEPRECATED !

import os
import os.path as op
import numpy as np
from pixell import enmap, utils



# import enlib if needed
try:
    from enlib import planet9
except ModuleNotFoundError:
    print("enlib not found, will not be able to mask planets")
    pass



def get_unprocessed_maps(map_path, odir, output="dirs", getall=False):
    # Bad maps
    badmaps = inputs.get_badmaps()

    dirs = os.listdir(map_path)
    # delete files that are not directories
    dirs = [d for d in dirs if op.isdir(op.join(map_path, d))]
    # sort dirs
    dirs.sort()

    maps_total = 0
    maps_processed = 0
    maps_unprocessed = 0
    subdir_unprocessed = []
    files_unprocessed = []
    for subdir in dirs:

        # list files with map extension in directory
        files = os.listdir(op.join(map_path, subdir))

        # only include files with map extension
        files = [f for f in files if f.endswith("map.fits")]

        # cut bad ctimes
        files = [f for f in files if f.split("_")[1] not in badmaps]

        # only files that have f220, f150, f090, f030, f040 in name
        files = [
            f
            for f in files
            if "f220" in f or "f150" in f or "f090" in f or "f040" in f or "f030" in f
        ]

        # Check if subdir is empty
        if len(files) == 0:
            continue

        for f in files:

            maps_total += 1

            # check if output file already exists
            if op.exists(op.join(odir, subdir, f[:-8] + "ocat.fits")):
                maps_processed += 1
                continue

            else:
                maps_unprocessed += 1
                subdir_unprocessed.append(subdir)
                if not getall:
                    files_unprocessed.append(op.join(map_path, subdir, f))
            if getall:
                files_unprocessed.append(op.join(map_path, subdir, f))

    if output == "dirs":
        return list(set(subdir_unprocessed))
    elif output == "files":
        return files_unprocessed




def depth1_pipeline(
    data: Depth1,
    grid: float,
    edgecut: int,
    galmask: enmap.ndmap,
    sourcemask=None,
    detection_threshold=5,
    sourcecut=5,
    ast_dir=None,
    ast_orbits=None,
    ast_tol=10.0 * utils.arcmin,
    verbosity=1,
):
    """
    Pipeline for transient detection

    Args:
        data: Depth1 object
        grid: gridsize in deg
        edgecut: number of pixels to cut from edge
        galmask: galaxy mask
        sourcemask: source catalog
        detection_threshold: snr threshold for initial source detection
        sourcecut: crossmatch radius for point source cut in arcminutes
        ast_dir: path to asteroid ephemeris directory
        ast_orbits: asteroid orbits
        ast_tol: tolerance for asteroid cut in arcmin
        verbosity: verbosity level

    Returns:
        astropy table of candidates
    """

    

    # check if map is all zeros
    if np.all(data.intensity_map == 0.0):
        if verbosity > 0:
            print("map is all zeros, skipping")

        return []

    # get galaxy, planet, edge masked flux, snr, tmap
    flux = data.masked_map(data.flux(), edgecut, galmask)
    snr = data.masked_map(data.snr(), edgecut, galmask)
    dflux = data.dflux()

    # renormalize maps using tile method
    med_ratio = get_medrat(
        snr,
        get_tmap_tiles(
            data.time_map,
            grid,
            snr,
            id=f"{data.map_ctime}_{data.wafer_name}_{data.freq}",
        ),
    )
    flux *= med_ratio
    snr *= med_ratio

    # initial point source detection
    ra_can, dec_can = source_detection(
        snr,
        flux,
        detection_threshold,
        verbosity=verbosity - 1,
    )
    if not ra_can:
        if verbosity > 0:
            print("did not find any candidates")
        return []
    if verbosity > 0:
        print("number of initial candidates: ", len(ra_can))

    # separate sources
    if sourcemask:
        sources = sourcemask[
            sourcemask["fluxJy"] > (inputs.get_sourceflux_threshold(data.freq) / 1000.0)
        ]
        catalog_match = tools.crossmatch_mask(
            np.array([dec_can, ra_can]).T,
            np.array([sources["decDeg"], sources["RADeg"]]).T,
            sourcecut,
        )
    else:
        catalog_match = np.array(len(ra_can) * [False])

    # mask blazar
    catalog_match = np.logical_or(
        catalog_match, blazar_crossmatch(np.array([dec_can, ra_can]).T)
    )

    if verbosity > 0:
        print(
            "number of candidates after point source/blazar cut: ",
            len(ra_can) - np.sum(catalog_match),
        )

    # matched filter and renormalization
    ra_new, dec_new, mf_sub, renorm_sub = perform_matched_filter(
        data, ra_can, dec_can, catalog_match, sourcemask, detection_threshold
    )
    if verbosity > 0:
        print("number of candidates after matched filter: ", np.sum(mf_sub))
        print("number of candidates after renormalization: ", np.sum(renorm_sub))

    coords = np.array(np.deg2rad([dec_new, ra_new]))
    source_string_names = [radec_to_str_name(ra_new[i], dec_new[i]) for i in range(len(ra_new))]
    print(source_string_names)
    
    flux_new = flux.at(coords, order=0, mask_nan=True)
    dflux_new = dflux.at(coords, order=0, mask_nan=True)
    snr_new = snr.at(coords, order=0, mask_nan=True)
    ctime_new = data.time_map.at(coords, order=0, mask_nan=True) + np.ones(len(ra_new)) * data.map_ctime
    source_candidates = []
    for i in range(len(ra_new)):
        source_candidates.append(
            SourceCandidate(ra=ra_new[i]%360,
                            dec=dec_new[i],
                            flux=flux_new[i],
                            dflux=dflux_new[i],
                            snr=snr_new[i],
                            ctime=ctime_new[i],
                            sourceID=source_string_names[i],
                            match_filtered=mf_sub[i],
                            renormalized=renorm_sub[i],
                            catalog_crossmatch=catalog_match[i],
                           )
        )
    
    # count sources
    source_count = len(source_candidates)

    '''
    # convert to pandas df
    ocat = pd.DataFrame.from_dict(source_candidates.dict())

    # asteroid cut
    if ast_dir:
        ocat, ast_cut = asteroid_cut(
            ocat, ast_dir, ast_tol, ast_orbits, verbosity=verbosity - 1
        )
    
    # convert to astropy table
    ocat = Table.from_pandas(ocat)

    # add units
    ocat["ra"].unit = u.deg
    ocat["dec"].unit = u.deg
    ocat["flux"].unit = u.mJy
    ocat["dflux"].unit = u.mJy
    ocat["ctime"].unit = u.s

    # add metadata
    ocat.meta["map_ctime"] = data.map_ctime
    ocat.meta["freq"] = data.freq
    ocat.meta["array"] = data.wafer_name
    ocat.meta["grid"] = grid
    ocat.meta["edgecut"] = edgecut
    ocat.meta["thresh"] = detection_threshold
    ocat.meta["ninit"] = len(ra_can)
    ocat.meta["mfcut"] = len(ra_can) - mfcount
    ocat.meta["nsource"] = source_count
    if ast_dir:
        ocat.meta["asttol"] = np.rad2deg(ast_tol) * 60
        ocat.meta["nast"] = ast_cut
    '''
    return source_candidates
    
