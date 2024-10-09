import os
import os.path as op
import math
import numpy as np
import pandas as pd
from scipy import spatial
from astropy.table import Table
from pixell import enmap, utils
import inputs, tools


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
                            [dec_deg * utils.degree, ra_deg * utils.degree],
                        ):
                            dist = tools.obj_dist(ivar, np.array([[dec_deg, ra_deg]]))[
                                0
                            ]
                            dec_pix, ra_pix = ivar.sky2pix(
                                [dec_deg * utils.degree, ra_deg * utils.degree]
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
                dist = utils.angdist(
                    np.array([ra_src, dec_src]) * utils.degree,
                    np.array([ra, dec]) * utils.degree,
                )
                fwhm = inputs.get_fwhm_arcmin(arr, freq)
                fwhm_deg = fwhm / 60.0
                pos_err = fwhm_deg / snr_src
                var_w += (dist**2.0) * (1 / pos_err**2.0)
        var_w /= invar
        pos_err_arcmin = var_w**0.5 * 60 / utils.degree
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
