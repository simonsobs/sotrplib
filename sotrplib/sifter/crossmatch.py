import os
import os.path as op
import math
import numpy as np
import pandas as pd
from scipy import spatial
from astropy.table import Table
from pixell import enmap
from pixell import utils as pixell_utils



def crossmatch_mask(sources, crosscat, radius:float,mode:str='all',return_matches:bool=False):
    """Determines if source matches with masked objects

    Args:
        sources: np.array of sources [[dec, ra]] in deg
        crosscat: catalog of masked objects [[dec, ra]] in deg
        radius: radius to search for matches in arcmin, float or list of length sources.
        mode: return `all` pairs, or just `closest`
    Returns:
        mask column for sources, 1 matches with at least one source, 0 no matches

    """

    crosspos_ra = crosscat[:, 1] * pixell_utils.degree
    crosspos_dec = crosscat[:, 0] * pixell_utils.degree
    crosspos = np.array([crosspos_ra, crosspos_dec]).T

    # if only one source
    if len(sources.shape) == 1:
        source_ra = sources[1] * pixell_utils.degree
        source_dec = sources[0] * pixell_utils.degree
        sourcepos = np.array([[source_ra, source_dec]])
        if isinstance(radius,(list,np.array)):
            r = radius[0] * pixell_utils.arcmin
        else:
            r = radius * pixell_utils.arcmin
        match = pixell_utils.crossmatch(sourcepos, crosspos, r, mode=mode)
        
        if len(match) > 0:
            return True if not return_matches else True,match
        else:
            return False if not return_matches else False,match

    mask = np.zeros(len(sources), dtype=bool)
    matches = []
    for i, _ in enumerate(mask):
        source_ra = sources[i, 1] * pixell_utils.degree
        source_dec = sources[i, 0] * pixell_utils.degree
        sourcepos = np.array([[source_ra, source_dec]])
        if isinstance(radius,(list,np.array)):
            r = radius[i] * pixell_utils.arcmin
        else:
            r = radius * pixell_utils.arcmin
        match = pixell_utils.crossmatch(sourcepos, crosspos, r, mode=mode)
        if len(match) > 0:
            mask[i] = True
        matches.append([(i,m[1]) for m in match])
    if return_matches:
        return mask,matches
    else:
        return mask

def sift(extracted_sources,
         catalog_sources,
         radius1Jy:float=30.0,
         min_match_radius:float=1.5,
         source_fluxes:list = None,
         map_freq:str = None,
         arr:str=None,
         fwhm_cut = 5.0,
         ):
    from ..utils.utils import radec_to_str_name
    from ..sources.sources import SourceCandidate

    """
     Perform crossmatching of extracted sources from `extract_sources` and the cataloged sources.
     Return lists of dictionaries containing each source which matches the catalog, may be noise, or appears to be a transient.

     Uses a flux-based matching radius with `radius1Jy` the radius, in arcmin, for a 1Jy source and 
     `min_match_radius` the radius, in arcmin, for a zero flux source, up to a max of 2 degrees.

     Args:
       extracted_sources:dict
           sources returned from extract_sources function
       catalog_sources:astropy table
           source catalog returned from load_act_catalog
       radius1Jy:float=30.0
           matching radius for a 1Jy source, arcmin
       min_match_radius:float=1.5
           minimum matching radius, i.e. for a zero flux source, arcmin
       source_fluxes:list = None,
           a list of the fluxes of the extracted sources, if None will pull it from `extracted_sources` dict. 
       fwhm_cut = 5.0
           a simple cut on fwhm, in arcmin, above which something is considered noise.
     Returns:
        source_candidates, transient_candidates, noise_candidates : list
            list of dictionaries with information about the detected source.

    """    

    if isinstance(source_fluxes,type(None)):
        source_fluxes = np.asarray([extracted_sources[f]['peakval']/1000. for f in extracted_sources])
    
    extracted_ra = np.asarray([extracted_sources[f]['ra'] for f in extracted_sources])
    extracted_dec = np.asarray([extracted_sources[f]['dec'] for f in extracted_sources])

    crossmatch_radius = np.minimum(np.maximum(source_fluxes*radius1Jy,min_match_radius),120)
    isin_cat,catalog_match = crossmatch_mask(np.asarray([extracted_dec/ pixell_utils.degree,extracted_ra/ pixell_utils.degree ]).T,
                                             np.asarray([catalog_sources["decDeg"], catalog_sources["RADeg"]]).T,
                                             list(crossmatch_radius),
                                             mode='closest',
                                             return_matches=True
                                            )


    source_candidates = []
    transient_candidates = []
    noise_candidates = []
    for source,cand_pos in enumerate(zip(extracted_ra,extracted_dec)):
        forced_photometry_info = extracted_sources[source]
        source_string_name = radec_to_str_name(cand_pos[0]/pixell_utils.degree,
                                               cand_pos[1]/pixell_utils.degree
                                              )
            
        if isin_cat[source]:
            crossmatch_name = catalog_sources['name'][catalog_match[source][0][1]]
        else:
            crossmatch_name = ''

        ## peakval is the max value within the kron radius while kron_flux is the integrated flux (i.e. assuming resolved)
        ## since filtered maps are convolved w beam, use max pixel value.
        cand = SourceCandidate(ra=cand_pos[0]/pixell_utils.degree%360,
                               dec=cand_pos[1]/pixell_utils.degree,
                               flux=forced_photometry_info['peakval'],
                               dflux=forced_photometry_info['peakval']/forced_photometry_info['peaksig'],
                               kron_flux=forced_photometry_info['kron_flux'],
                               kron_fluxerr=forced_photometry_info['kron_fluxerr'],
                               kron_radius = forced_photometry_info['kron_radius'],
                               snr=forced_photometry_info['kron_flux']/forced_photometry_info['kron_fluxerr'],
                               freq=str(map_freq),
                               arr=arr,
                               ctime=forced_photometry_info['time'],
                               sourceID=source_string_name,
                               match_filtered=False,
                               renormalized=True,
                               catalog_crossmatch=isin_cat[source],
                               crossmatch_name=crossmatch_name,
                               ellipticity=forced_photometry_info['ellipticity'],
                               elongation=forced_photometry_info['elongation'],
                               fwhm=forced_photometry_info['fwhm']
                              )
        ## do sifting operations here...
        if not np.isfinite(forced_photometry_info['kron_flux']) or not np.isfinite(forced_photometry_info['kron_fluxerr']) or cand.fwhm>=fwhm_cut or not np.isfinite(forced_photometry_info['peakval']) :
            noise_candidates.append(cand)
        elif isin_cat[source]:
            source_candidates.append(cand)
        else:
            transient_candidates.append(cand)
        del cand
        
    print(len(source_candidates),'catalog matches')
    print(len(transient_candidates),'transient candidates')
    print(len(noise_candidates),'probably noise') 
    return source_candidates, transient_candidates, noise_candidates


####################
## Act crossmatching between wafers / bands etc.
####################

def get_cats(
    path, ctime, return_fnames=False, mfcut=True, renromcut=False, sourcecut=True
):
    keys = [
        "pa4_f220",
        "pa4_f150",
        "pa5_f150",
        "pa5_f090",
        "pa6_f150",
        "pa6_f090",
        "pa7_f040",
        "pa7_f030",
    ]
    cats = []
    fnames = []
    for i, key in enumerate(keys):
        f = op.join(path, f"depth1_{ctime}_{key}_ocat.fits")

        # check if file exists. If not, append empty array
        if not op.exists(f):
            cats.append(
                pd.DataFrame({"ra": [], "dec": [], "flux": [], "snr": [], "ctime": []})
            )
            fnames.append(np.nan)
            continue

        fname = op.join(path, f)
        t = Table.read(fname)
        t = t.to_pandas()
        if len(t) != 0:
            if mfcut:
                t = t[t["mf"].astype(bool)]
            if renromcut:
                t = t[t["renorm"].astype(bool)]
            if sourcecut:
                t = t[~t["source"].astype(bool)]
        # reindex
        t.index = np.arange(len(t))
        cats.append(t)
        fnames.append(fname)

    cats = dict(zip(keys, cats))

    if return_fnames:
        return cats, fnames

    else:
        return cats


def crossmatch_array(cats_in_order, N, radius):
    """crossmatch btw 6 catalogs to get a matched catalog with inds from individual cats, the matched candidate should be detected by at least N arrays

    Args:
        cats_in_order: a list of 2D np.array of source, each with [[dec,ra]] in deg, order is pa4_f220,pa4_f150, pa5_f150,pa5f090,pa6f150,pa6f090. For non-existin cats, put an empty 2D array in the position in list
        N: integer that candidate should be detected by at least N arrays
        radius: radius to search for matches in arcmin
    """
    arr_names = ["pa4", "pa4", "pa5", "pa5", "pa6", "pa6", "pa7", "pa7"]
    keys = [
        "pa4_f220",
        "pa4_f150",
        "pa5_f150",
        "pa5_f090",
        "pa6_f150",
        "pa6_f090",
        "pa7_f040",
        "pa7_f030",
    ]
    rad_deg = radius / 60.0
    skip = 0
    cat_full = []
    for i, cat in enumerate(cats_in_order):
        if cat.shape[0] == 0:
            skip += 1
            continue
        else:
            patch1 = np.arange(0, cat.shape[0], dtype=int)[np.newaxis].T
            patch2 = np.ones(shape=(cat.shape[0], 1)) * i  # i for index of the cats
            cat = np.concatenate((cat, patch1), axis=1)
            cat = np.concatenate((cat, patch2), axis=1)
            if i - skip == 0:
                cat_full = cat
            else:
                cat_full = np.concatenate((cat_full, cat), axis=0)
    del cat
    if len(cat_full) == 0:
        new_cat = np.zeros((0, 8))
    else:
        tree = spatial.cKDTree(cat_full[:, 0:2])
        idups_raw = tree.query_ball_tree(tree, rad_deg)
        new_cat = []
        idups_merge = []
        for i, srcs in enumerate(idups_raw):
            if i == 0:
                idups_merge.append(srcs)
            else:
                overlap = 0
                for srcs2 in idups_merge:
                    if bool(set(srcs) & set(srcs2)):
                        srcs_m = list(set(srcs + srcs2))
                        idups_merge.remove(srcs2)
                        idups_merge.append(srcs_m)
                        overlap = 1
                if overlap == 0:
                    idups_merge.append(srcs)
        idups_redu = []
        for i, srcs in enumerate(idups_merge):
            inds = cat_full[srcs, -1].astype(int)
            arrs = np.unique(np.array(arr_names)[inds])
            if arrs.shape[0] > N - 1 and srcs not in idups_redu:
                row = np.ones(8) * float("nan")
                for src in srcs:
                    ind = int(cat_full[src, -1])
                    row[ind] = cat_full[src, -2]
                new_cat.append(row)
                idups_redu.append(srcs)

    new_cat = np.array(new_cat)
    if len(new_cat) == 0:
        new_cat = np.zeros((0, 8))
    ocat_dtype = [
        ("pa4_f220", "1f"),
        ("pa4_f150", "1f"),
        ("pa5_f150", "1f"),
        ("pa5_f090", "1f"),
        ("pa6_f150", "1f"),
        ("pa6_f090", "1f"),
        ("pa7_f040", "1f"),
        ("pa7_f030", "1f"),
    ]
    ocat = np.zeros(len(new_cat), ocat_dtype).view(np.recarray)
    ocat.pa4_f220 = new_cat[:, 0]
    ocat.pa4_f150 = new_cat[:, 1]
    ocat.pa5_f150 = new_cat[:, 2]
    ocat.pa5_f090 = new_cat[:, 3]
    ocat.pa6_f150 = new_cat[:, 4]
    ocat.pa6_f090 = new_cat[:, 5]
    ocat.pa7_f040 = new_cat[:, 6]
    ocat.pa7_f030 = new_cat[:, 7]
    return ocat


def crossmatch_array_new(cats_in_order, N, radius, ctime):
    """crossmatch btw 8 catalogs to get a matched catalog with inds from individual cats, the matched candidate should be detected by at least N arrays
       but it would still keep the catalog crossmatched by two bands if there's only 1 array's data available

    Args:
        cats_in_order: a list of 2D np.array of source, each with [[dec,ra]] in deg, order is pa4_f220,pa4_f150, pa5_f150,pa5f090,pa6f150,pa6f090. For non-existin cats, put an empty 2D array in the position in list
        N: integer that candidate should be detected by at least N arrays
        radius: radius to search for matches in arcmin
    """
    from ..utils.utils import obj_dist
    arr_names = ["pa4", "pa4", "pa5", "pa5", "pa6", "pa6", "pa7", "pa7"]
    keys = [
        "pa4_f220",
        "pa4_f150",
        "pa5_f150",
        "pa5_f090",
        "pa6_f150",
        "pa6_f090",
        "pa7_f040",
        "pa7_f030",
    ]
    rad_deg = radius / 60.0
    skip = 0
    cat_full = []
    for i, cat in enumerate(cats_in_order):
        if cat.shape[0] == 0:
            skip += 1
            continue
        else:
            patch1 = np.arange(0, cat.shape[0], dtype=int)[np.newaxis].T
            patch2 = np.ones(shape=(cat.shape[0], 1)) * i  # i for index of the cats
            cat = np.concatenate((cat, patch1), axis=1)
            cat = np.concatenate((cat, patch2), axis=1)
            if i - skip == 0:
                cat_full = cat
            else:
                cat_full = np.concatenate((cat_full, cat), axis=0)
    del cat
    if len(cat_full) == 0:
        new_cat = np.zeros((0, 8))
    else:
        tree = spatial.cKDTree(cat_full[:, 0:2])
        idups_raw = tree.query_ball_tree(tree, rad_deg)
        new_cat = []
        idups_merge = []
        for i, srcs in enumerate(idups_raw):
            if i == 0:
                idups_merge.append(srcs)
            else:
                overlap = 0
                for srcs2 in idups_merge:
                    if bool(set(srcs) & set(srcs2)):
                        srcs_m = list(set(srcs + srcs2))
                        idups_merge.remove(srcs2)
                        idups_merge.append(srcs_m)
                        overlap = 1
                if overlap == 0:
                    idups_merge.append(srcs)
        idups_redu = []
        for i, srcs in enumerate(idups_merge):
            inds = cat_full[srcs, -1].astype(int)
            arrs = np.unique(np.array(arr_names)[inds])
            keys_srcs = np.unique(np.array(keys)[inds])
            if arrs.shape[0] > N - 1 and srcs not in idups_redu:
                row = np.ones(8) * float("nan")
                for src in srcs:
                    ind = int(cat_full[src, -1])
                    row[ind] = cat_full[src, -2]
                new_cat.append(row)
                idups_redu.append(srcs)
            # check if all other freq arr data is not available
            elif arrs.shape[0] == 1 and keys_srcs.shape[0] == 2:
                src = srcs[0]
                ra_deg = cat_full[src, 0]
                dec_deg = cat_full[src, 1]
                keys_left = [key for key in keys if key not in keys_srcs]
                count_ava = 0
                for key in keys_left:
                    ivar_file = (
                        "/scratch/gpfs/snaess/actpol/maps/depth1/release/%s/depth1_%s_%s_ivar.fits"
                        % (ctime[:5], ctime, key)
                    )
                    if os.path.exists(ivar_file) == True:
                        ivar = enmap.read_map(ivar_file)
                        if enmap.contains(
                            ivar.shape,
                            ivar.wcs,
                            [dec_deg * pixell_utilsdegree, ra_deg * pixell_utilsdegree],
                        ):
                            dist = obj_dist(ivar, np.array([[dec_deg, ra_deg]]))[
                                0
                            ]
                            dec_pix, ra_pix = ivar.sky2pix(
                                [dec_deg * pixell_utilsdegree, ra_deg * pixell_utilsdegree]
                            )
                            ivar_src = ivar[int(dec_pix), int(ra_pix)]
                            if ivar_src != 0.0 and dist > 10.0:
                                count_ava += 1
                if count_ava == 0 and srcs not in idups_redu:
                    row = np.ones(8) * float("nan")
                    for src in srcs:
                        ind = int(cat_full[src, -1])
                        row[ind] = cat_full[src, -2]
                    new_cat.append(row)
                    idups_redu.append(srcs)
    new_cat = np.array(new_cat)
    if len(new_cat) == 0:
        new_cat = np.zeros((0, 8))
    ocat_dtype = [
        ("pa4_f220", "1f"),
        ("pa4_f150", "1f"),
        ("pa5_f150", "1f"),
        ("pa5_f090", "1f"),
        ("pa6_f150", "1f"),
        ("pa6_f090", "1f"),
        ("pa7_f040", "1f"),
        ("pa7_f030", "1f"),
    ]
    ocat = np.zeros(len(new_cat), ocat_dtype).view(np.recarray)
    ocat.pa4_f220 = new_cat[:, 0]
    ocat.pa4_f150 = new_cat[:, 1]
    ocat.pa5_f150 = new_cat[:, 2]
    ocat.pa5_f090 = new_cat[:, 3]
    ocat.pa6_f150 = new_cat[:, 4]
    ocat.pa6_f090 = new_cat[:, 5]
    ocat.pa7_f040 = new_cat[:, 6]
    ocat.pa7_f030 = new_cat[:, 7]
    return ocat


def crossmatch_freq_array(cats_in_order, N, radius):
    """crossmatch btw 6 catalogs to get a matched catalog with inds from individual cats, the matched candidate should be detected by at least N arrays

    Args:
        cats_in_order: a list of 2D np.array of source, each with [[dec,ra]] in deg, order is pa4_f220,pa4_f150, pa5_f150,pa5f090,pa6f150,pa6f090. For non-existin cats, put an empty 2D array in the position in list
        N: integer that candidate should be detected by at least N arrays
        radius: radius to search for matches in arcmin
    """
    arr_names = ["pa4", "pa4", "pa5", "pa5", "pa6", "pa6", "pa7", "pa7"]
    keys = [
        "pa4_f220",
        "pa4_f150",
        "pa5_f150",
        "pa5_f090",
        "pa6_f150",
        "pa6_f090",
        "pa7_f040",
        "pa7_f030",
    ]
    rad_deg = radius / 60.0
    skip = 0
    cat_full = []
    for i, cat in enumerate(cats_in_order):
        if cat.shape[0] == 0:
            skip += 1
            continue
        else:
            patch1 = np.arange(0, cat.shape[0], dtype=int)[np.newaxis].T
            patch2 = np.ones(shape=(cat.shape[0], 1)) * i  # i for index of the cats
            cat = np.concatenate((cat, patch1), axis=1)
            cat = np.concatenate((cat, patch2), axis=1)
            if i - skip == 0:
                cat_full = cat
            else:
                cat_full = np.concatenate((cat_full, cat), axis=0)
    del cat
    if len(cat_full) == 0:
        new_cat = np.zeros((0, 8))
    else:
        tree = spatial.cKDTree(cat_full[:, 0:2])
        idups_raw = tree.query_ball_tree(tree, rad_deg)
        new_cat = []
        idups_merge = []
        for i, srcs in enumerate(idups_raw):
            if i == 0:
                idups_merge.append(srcs)
            else:
                overlap = 0
                for srcs2 in idups_merge:
                    if bool(set(srcs) & set(srcs2)):
                        srcs_m = list(set(srcs + srcs2))
                        idups_merge.remove(srcs2)
                        idups_merge.append(srcs_m)
                        overlap = 1
                if overlap == 0:
                    idups_merge.append(srcs)
        idups_redu = []
        for i, srcs in enumerate(idups_merge):
            inds = cat_full[srcs, -1].astype(int)
            keys_srcs = np.unique(np.array(keys)[inds])
            if keys_srcs.shape[0] > N - 1 and srcs not in idups_redu:
                row = np.ones(8) * float("nan")
                for src in srcs:
                    ind = int(cat_full[src, -1])
                    row[ind] = cat_full[src, -2]
                new_cat.append(row)
                idups_redu.append(srcs)

    new_cat = np.array(new_cat)
    if len(new_cat) == 0:
        new_cat = np.zeros((0, 8))
    ocat_dtype = [
        ("pa4_f220", "1f"),
        ("pa4_f150", "1f"),
        ("pa5_f150", "1f"),
        ("pa5_f090", "1f"),
        ("pa6_f150", "1f"),
        ("pa6_f090", "1f"),
        ("pa7_f040", "1f"),
        ("pa7_f030", "1f"),
    ]
    ocat = np.zeros(len(new_cat), ocat_dtype).view(np.recarray)
    ocat.pa4_f220 = new_cat[:, 0]
    ocat.pa4_f150 = new_cat[:, 1]
    ocat.pa5_f150 = new_cat[:, 2]
    ocat.pa5_f090 = new_cat[:, 3]
    ocat.pa6_f150 = new_cat[:, 4]
    ocat.pa6_f090 = new_cat[:, 5]
    ocat.pa7_f040 = new_cat[:, 6]
    ocat.pa7_f030 = new_cat[:, 7]
    return ocat


def crossmatch_col(cats, cross_idx):
    new_cats = []
    keys = [
        "pa4_f220",
        "pa4_f150",
        "pa5_f150",
        "pa5_f090",
        "pa6_f150",
        "pa6_f090",
        "pa7_f040",
        "pa7_f030",
    ]
    for i, df in enumerate(cats):
        indices = cross_idx[keys[i]]
        indices = [x for x in indices if str(x) != "nan"]
        df["crossmatch"] = False
        df.loc[indices, "crossmatch"] = True
        new_cats.append(df)
    return new_cats


def merge_cats(full_cats_in_order, N, radius, saveps=False, type="arr", ctime=None):
    """merge btw 6 catalogs to get one single merged catalog, with on overall ra, dec, position error

    Args:
        full_cats_in_order: fits.open(file)[1].data from fits file, order is pa4_f220,pa4_f150, pa5_f150,pa5f090,pa6f150,pa6f090.
        N: integer that candidate should be detected by at least N arrays
        radius: radius to search for matches in arcmin
        saveps: if True, output point sources
        type: arr if crossmatch by array, freq if crossmatch by freq or arr, cond if crossmatch by array but use freq if only one array's data is available
        ctime: ctime in the file name with string format, needed if type is cond
    """
    cats_pos = []
    for i in range(8):
        cat_data = full_cats_in_order[i]
        cat_array = np.zeros((cat_data.shape[0], 2))
        cat_array[:, 0] = cat_data.ra
        cat_array[:, 1] = cat_data.dec
        cats_pos.append(cat_array)
    if type == "arr":
        matched_inds = crossmatch_array(cats_pos, N, radius)
    elif type == "freq":
        matched_inds = crossmatch_freq_array(cats_pos, N, radius)
    elif type == "cond":
        matched_inds = crossmatch_array_new(cats_pos, N, radius, ctime)
    else:
        raise ValueError("type must be arr, freq, or cond")
    new_cats = crossmatch_col(full_cats_in_order, matched_inds)
    cat_merge = []
    for i in range(matched_inds.shape[0]):
        ra = 0
        dec = 0
        invar = 0
        count = 0
        ps = 0
        for j, [arr, freq] in enumerate(
            [
                ["pa4", "f220"],
                ["pa4", "f150"],
                ["pa5", "f150"],
                ["pa5", "f090"],
                ["pa6", "f150"],
                ["pa6", "f090"],
                ["pa7", "f040"],
                ["pa7", "f030"],
            ]
        ):
            key = "%s_%s" % (arr, freq)
            if not math.isnan(matched_inds[key][i]):
                fwhm = inputs.get_fwhm_arcmin(arr, freq)
                fwhm_deg = fwhm / 60.0
                idx_src = int(matched_inds[key][i])
                cat_src = full_cats_in_order[j]
                ps += cat_src.source[idx_src]
                ra_src = cat_src.ra[idx_src]
                dec_src = cat_src.dec[idx_src]
                snr_src = cat_src.snr[idx_src]
                pos_err = fwhm_deg / snr_src
                ra += ra_src / pos_err**2.0
                dec += dec_src / pos_err**2.0
                invar += 1 / pos_err**2.0
        if invar == 0:
            print("ZeroDivisionError: Candidate Removed")
            continue
        ra /= invar
        dec /= invar
        var_w = 0
        if (ps > 0) and not saveps:
            continue
        for j, [arr, freq] in enumerate(
            [
                ["pa4", "f220"],
                ["pa4", "f150"],
                ["pa5", "f150"],
                ["pa5", "f090"],
                ["pa6", "f150"],
                ["pa6", "f090"],
                ["pa7", "f040"],
                ["pa7", "f030"],
            ]
        ):
            key = "%s_%s" % (arr, freq)
            if not math.isnan(matched_inds[key][i]):
                idx_src = int(matched_inds[key][i])
                ra_src = full_cats_in_order[j].ra[idx_src]
                dec_src = full_cats_in_order[j].dec[idx_src]
                dist = pixell_utilsangdist(
                    np.array([ra_src, dec_src]) * pixell_utilsdegree,
                    np.array([ra, dec]) * pixell_utilsdegree,
                )
                fwhm = inputs.get_fwhm_arcmin(arr, freq)
                fwhm_deg = fwhm / 60.0
                pos_err = fwhm_deg / snr_src
                var_w += (dist**2.0) * (1 / pos_err**2.0)
        var_w /= invar
        pos_err_arcmin = var_w**0.5 * 60 / pixell_utilsdegree
        cat_merge.append(
            np.array(
                [
                    ra,
                    dec,
                    pos_err_arcmin,
                    matched_inds["pa4_f220"][i],
                    matched_inds["pa4_f150"][i],
                    matched_inds["pa5_f150"][i],
                    matched_inds["pa5_f090"][i],
                    matched_inds["pa6_f150"][i],
                    matched_inds["pa6_f090"][i],
                    matched_inds["pa7_f040"][i],
                    matched_inds["pa7_f030"][i],
                ]
            )
        )
    if len(cat_merge) == 0:
        cat_merge = np.zeros((0, 11))
    else:
        cat_merge = np.vstack(cat_merge)
    ocat_dtype = [
        ("ra", "1f"),
        ("dec", "1f"),
        ("pos_err_arcmin", "1f"),
        ("pa4_f220", "1f"),
        ("pa4_f150", "1f"),
        ("pa5_f150", "1f"),
        ("pa5_f090", "1f"),
        ("pa6_f150", "1f"),
        ("pa6_f090", "1f"),
        ("pa7_f040", "1f"),
        ("pa7_f030", "1f"),
    ]
    ocat = np.zeros(len(cat_merge), ocat_dtype).view(np.recarray)
    ocat.ra = cat_merge[:, 0]
    ocat.dec = cat_merge[:, 1]
    ocat.pos_err_arcmin = cat_merge[:, 2]
    ocat.pa4_f220 = cat_merge[:, 3]
    ocat.pa4_f150 = cat_merge[:, 4]
    ocat.pa5_f150 = cat_merge[:, 5]
    ocat.pa5_f090 = cat_merge[:, 6]
    ocat.pa6_f150 = cat_merge[:, 7]
    ocat.pa6_f090 = cat_merge[:, 8]
    ocat.pa7_f040 = cat_merge[:, 9]
    ocat.pa7_f030 = cat_merge[:, 10]
    return ocat, new_cats


def candidate_output(
    cross_ocat: pd.DataFrame,
    cats: dict,
    odir: str,
    ctime: float,
    verbosity=0,
    overwrite=True,
):
    # for each candidate, write output
    keys = list(cats.keys())
    for i in range(len(cross_ocat)):
        ocat = {
            "idx": [],
            "band": [],
            "ra": [],
            "dec": [],
            "flux": [],
            "dflux": [],
            "snr": [],
            "ctime": [],
        }

        # create output dir
        can_odir = op.join(odir, f"{ctime}_{str(i)}")
        if not op.exists(can_odir):
            os.makedirs(can_odir)

        # get ra, dec, flux, dflux, snr, ctime for each candidate
        for k in keys:
            # check if detection exists
            if np.isnan(cross_ocat[k][i]):
                continue

            # get ra, dec, flux, dflux, snr, ctime
            ocat["idx"].append(cross_ocat[k][i])
            ocat["band"].append(k)
            ocat["ra"].append(cats[k]["ra"][int(cross_ocat[k][i])])
            ocat["dec"].append(cats[k]["dec"][int(cross_ocat[k][i])])
            ocat["flux"].append(cats[k]["flux"][int(cross_ocat[k][i])])
            ocat["dflux"].append(cats[k]["dflux"][int(cross_ocat[k][i])])
            ocat["snr"].append(cats[k]["snr"][int(cross_ocat[k][i])])
            ocat["ctime"].append(cats[k]["ctime"][int(cross_ocat[k][i])])

        # Convert to table
        output = pd.DataFrame(ocat)
        output = Table.from_pandas(output)

        # metadata
        output.meta["ra_deg"] = cross_ocat["ra"][i]
        output.meta["dec_deg"] = cross_ocat["dec"][i]
        output.meta["err_am"] = cross_ocat["pos_err_arcmin"][i]
        output.meta["maptime"] = ctime

        # write output
        output.write(
            op.join(can_odir, f"{ctime}_{str(i)}_ocat.fits"), overwrite=overwrite
        )

        if verbosity > 0:
            print(f'Wrote {op.join(can_odir, f"{ctime}_{str(i)}_ocat.fits")}')

    return


def blazar_crossmatch(pos, tol=0.5):
    """
    Mask blazar 3C 454.3 using icrs position from SIMBAD

    Args:
        pos:  np.array of positions [[dec, ra]] in deg
        tol: tolerance in deg

    Returns:
        1 in blazar, 0 if not in blazar
    """
    blazar_pos = SkyCoord("22h53m57.7480438728s", "+16d08m53.561508864s", frame="icrs")
    can_pos = SkyCoord(pos[:, 1], pos[:, 0], frame="icrs", unit="deg")
    sep = blazar_pos.separation(can_pos).to(u.deg)
    match = []
    for s in sep:
        if s.value < tol:
            match.append(1)
        else:
            match.append(0)
    return np.array(match).astype(bool)