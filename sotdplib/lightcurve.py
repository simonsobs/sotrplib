## Lightcurve needs rewritten.

import os
import os.path as op
import numpy as np
import glob as gl
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from pixell import enmap, utils
from astropy.io import fits
from maps import kappa_clean, get_snr, extract_sources
from . import tools, filters, inputs



def lightcurve(
    coords,
    start=None,
    end=None,
    frequencies=None,
    sourcedetection=False,
    verbosity=0,
    mpi=False,
    fluxdir=None,
    mfdir=None,
    overwrite=False,
    masksources=True,
    pol=False,
):
    """get depth1 lightcurve

    Args:
        coords: coordinates of object [dec, ra] in radians
        start: start time in ctime, if None, start at beginning of data
        end: end time in ctime, if None, end at end of data
        frequencies: list of frequencies to include ['f090', 'f150', 'f220']
        sourcedetection: if True, use center of mass by flux to get position
        verbosity: verbosity level
        mpi: if True, run with mpi
        fluxdir: if not None, save flux maps to this directory
        mfdir: if not None, save matched filter maps to this directory

    Returns:
        lightcurve of object
    """

    if frequencies is None:
        frequencies = ["f030", "f040", "f090", "f150", "f220"]
    if mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    sourcecats = inputs.get_sourcecats()
    for f in frequencies:
        sourcecats[f] = sourcecats[f][
            sourcecats[f]["fluxJy"] > (inputs.get_sourceflux_threshold(f) / 1000.0)
        ]

    if fluxdir and rank == 0:
        if not op.exists(fluxdir):
            os.makedirs(fluxdir)
    if mfdir and rank == 0:
        if not op.exists(mfdir):
            os.makedirs(mfdir)

    if pol:
        light_curve = {
            "freq": [],
            "array": [],
            "ra": [],
            "dec": [],
            "ctime": [],
            "snr": [],
            "flux": [],
            "dflux": [],
            "Q": [],
            "U": [],
            "dQ": [],
            "dU": [],
            "detect": [],
        }
    else:
        light_curve = {
            "freq": [],
            "array": [],
            "ra": [],
            "dec": [],
            "ctime": [],
            "snr": [],
            "flux": [],
            "dflux": [],
            "detect": [],
        }

    for freq_label in frequencies:
        if freq_label == "f030":
            array_list = ["pa7"]
        elif freq_label == "f040":
            array_list = ["pa7"]
        elif freq_label == "f090":
            array_list = ["pa5", "pa6"]
        elif freq_label == "f150":
            array_list = ["pa4", "pa5", "pa6"]
        elif freq_label == "f220":
            array_list = ["pa4"]
        for array_label in array_list:
            if verbosity > 0 and rank == 0:
                print("Processing freq:%s array:%s " % (freq_label, array_label))
            ctimes_depth1 = np.loadtxt(
                "/scratch/gpfs/ccaimapo/Transients/data/Depth-1_maps/ctimes_%s_%s.txt"
                % (array_label, freq_label)
            )
            boxes_depth1 = np.loadtxt(
                "/scratch/gpfs/ccaimapo/Transients/data/Depth-1_maps/boxes_%s_%s.txt"
                % (array_label, freq_label)
            )  # boxes is ctime_ ra1 ra2 dec1 dec2 in radians

            # only use ctimes between start and end if they exist
            if start and end:
                mask = np.where(
                    np.logical_and(ctimes_depth1 >= start, ctimes_depth1 <= end)
                )
            elif start:
                mask = np.where(ctimes_depth1 >= start)
            elif end:
                mask = np.where(ctimes_depth1 <= end)
            else:
                mask = np.ones(len(ctimes_depth1), dtype=bool)

            ctimes_depth1 = ctimes_depth1[mask]
            boxes_depth1 = boxes_depth1[mask]

            Ntimes = len(ctimes_depth1)
            arr = np.arange(Ntimes)

            if mpi:
                if rank == 0:
                    arr = np.array_split(arr, size)
                else:
                    arr = None
                arr = comm.scatter(arr, root=0)

            for n_time in arr:
                ctime_ = ctimes_depth1[n_time]

                # mask with the sources that are inside the rectangle of the depth-1 map
                mask = np.logical_and(
                    utils.between_angles(
                        coords[1],
                        [boxes_depth1[n_time, 2], boxes_depth1[n_time, 1]],
                        period=2 * np.pi,
                    ),
                    np.logical_and(
                        boxes_depth1[n_time, 3] <= coords[0],
                        coords[0] <= boxes_depth1[n_time, 4],
                    ),
                )

                # if no sources in map, continue
                if mask.sum() == 0.0:
                    continue

                # otherwise there is at least 1 source in the depth 1 map
                else:
                    shape_full, wcs_full = enmap.read_map_geometry(
                        "/scratch/gpfs/snaess/actpol/maps/depth1/release/%s/depth1_%i_%s_%s_rho.fits"
                        % (str(ctime_)[0:5], ctime_, array_label, freq_label)
                    )
                    pixbox = enmap.neighborhood_pixboxes(
                        shape_full, wcs_full, coords.T, 0.5 * utils.degree
                    )
                    pos = np.array((coords[0], coords[1]))
                    # load the rho and kappa map
                    stamp_ivar = enmap.read_map(
                        "/scratch/gpfs/snaess/actpol/maps/depth1/release/%s/depth1_%i_%s_%s_ivar.fits"
                        % (str(ctime_)[0:5], ctime_, array_label, freq_label),
                        pixbox=pixbox,
                    )
                    if stamp_ivar.at(pos, order=0) == 0.0:
                        # test if the map
                        # this means the source is not observed in the depth-1 map, we continue
                        continue
                    else:
                        if verbosity > 1:
                            print("We have a measured flux at ctime = %i" % ctime_)
                        stamp_rho, stamp_q, stamp_u = enmap.read_map(
                            "/scratch/gpfs/snaess/actpol/maps/depth1/release/%s/depth1_%i_%s_%s_rho.fits"
                            % (str(ctime_)[0:5], ctime_, array_label, freq_label),
                            pixbox=pixbox,
                        )
                        stamp_kappa, stamp_kq, stamp_ku = enmap.read_map(
                            "/scratch/gpfs/snaess/actpol/maps/depth1/release/%s/depth1_%i_%s_%s_kappa.fits"
                            % (str(ctime_)[0:5], ctime_, array_label, freq_label),
                            pixbox=pixbox,
                        )
                        stamp_kappa = kappa_clean(stamp_kappa, stamp_rho)
                        stamp_time = enmap.read_map(
                            "/scratch/gpfs/snaess/actpol/maps/depth1/release/%s/depth1_%i_%s_%s_time.fits"
                            % (str(ctime_)[0:5], ctime_, array_label, freq_label),
                            pixbox=pixbox,
                        )
                        flux = stamp_rho / stamp_kappa
                        snr = get_snr(stamp_rho, stamp_kappa)
                        dflux = stamp_kappa**-0.5
                        if pol:
                            flux_Q = stamp_q / stamp_kq
                            flux_U = stamp_u / stamp_ku
                            dflux_Q = stamp_kq**-0.5
                            dflux_U = stamp_ku**-0.5

                        # source detection
                        detection = True
                        if sourcedetection:
                            cand_ra, cand_dec = source_detection(
                                snr, flux, threshold=3.0
                            )

                            # crossmatch with source catalog
                            if masksources:
                                if len(cand_ra) > 0:
                                    sources = sourcecats[freq_label]
                                    source_masking = tools.crossmatch_mask(
                                        np.array([cand_ra, cand_dec]).T,
                                        np.array(
                                            [sources["decDeg"], sources["RADeg"]]
                                        ).T,
                                        5.0,
                                    )

                                    cand_ra = np.array(cand_ra)
                                    cand_dec = np.array(cand_dec)

                                    cand_ra = cand_ra[~source_masking]
                                    cand_dec = cand_dec[~source_masking]

                            # Use new position if there is a detection
                            if len(cand_ra) > 0:
                                # Take closer detection if more than one
                                if len(cand_ra) > 1:
                                    sep = tools.sky_sep(
                                        cand_ra,
                                        cand_dec,
                                        np.rad2deg(pos[1]),
                                        np.rad2deg(pos[0]),
                                    )
                                    ra_new = cand_ra[np.argmin(sep)]
                                    dec_new = cand_dec[np.argmin(sep)]
                                    pos_new = np.deg2rad(np.array([dec_new, ra_new]))
                                else:
                                    pos_new = np.deg2rad(
                                        np.array([cand_dec[0], cand_ra[0]])
                                    )
                                detect_sep = tools.sky_sep(
                                    np.rad2deg(pos_new[1]),
                                    np.rad2deg(pos_new[0]),
                                    np.rad2deg(pos[1]),
                                    np.rad2deg(pos[0]),
                                )
                                if detect_sep * 60.0 > 2.0:
                                    detection = False
                                    if verbosity > 1:
                                        print(
                                            f"Object Detected, but separation of {detect_sep * 60.:.2f} arcmin is too large \n"
                                        )
                                else:
                                    if verbosity > 1:
                                        print(
                                            f"Object Detected, separation of {detect_sep * 60.:.2f} arcmin \n"
                                        )
                                    pos = pos_new

                            # If no detection, get upper limit
                            else:
                                if verbosity > 1:
                                    print("No detection \n")
                                detection = False
                        else:
                            detection = False

                        light_curve["freq"].append(freq_label)
                        light_curve["array"].append(array_label)
                        light_curve["ra"].append(pos[1])
                        light_curve["dec"].append(pos[0])
                        light_curve["ctime"].append(
                            ctime_ + stamp_time.at(pos, order=0, mask_nan=True)
                        )
                        light_curve["snr"].append(snr.at(pos, order=1, mask_nan=True))
                        light_curve["flux"].append(flux.at(pos, order=1, mask_nan=True))
                        light_curve["dflux"].append(
                            dflux.at(pos, order=1, mask_nan=True)
                        )
                        light_curve["detect"].append(detection)
                        if pol:
                            light_curve["Q"].append(
                                flux_Q.at(pos, order=1, mask_nan=True)
                            )
                            light_curve["U"].append(
                                flux_U.at(pos, order=1, mask_nan=True)
                            )
                            light_curve["dQ"].append(
                                dflux_Q.at(pos, order=1, mask_nan=True)
                            )
                            light_curve["dU"].append(
                                dflux_U.at(pos, order=1, mask_nan=True)
                            )
                        if fluxdir:
                            # check if flux map exists
                            if (
                                op.exists(
                                    op.join(
                                        fluxdir,
                                        f"{array_label}_{freq_label}_{int(ctime_)}_flux.fits",
                                    )
                                )
                                and not overwrite
                            ):
                                if verbosity > 0:
                                    print(
                                        f"Skipping {array_label}_{freq_label}_{int(ctime_)}_flux.fits because it already exists"
                                    )
                            else:
                                enmap.write_map(
                                    op.join(
                                        fluxdir,
                                        f"{array_label}_{freq_label}_{int(ctime_)}_flux.fits",
                                    ),
                                    flux,
                                )
                        if mfdir:
                            # check if mf map exists
                            if (
                                op.exists(
                                    op.join(
                                        mfdir,
                                        f"{array_label}_{freq_label}_{int(ctime_)}_mf.fits",
                                    )
                                )
                                and not overwrite
                            ):
                                if verbosity > 0:
                                    print(
                                        f"Skipping {array_label}_{freq_label}_{int(ctime_)}_mf.fits because it already exists"
                                    )
                            else:
                                ra = np.deg2rad(pos[1])
                                dec = np.deg2rad(pos[0])
                                stamp_imap = enmap.read_map(
                                    "/scratch/gpfs/snaess/actpol/maps/depth1/release/%s/depth1_%i_%s_%s_map.fits"
                                    % (
                                        str(ctime_)[0:5],
                                        ctime_,
                                        array_label,
                                        freq_label,
                                    ),
                                    pixbox=pixbox,
                                )[0]
                                rho_mf, kappa_mf = filters.matched_filter(
                                    stamp_imap,
                                    stamp_ivar,
                                    array_label,
                                    freq_label,
                                    ra,
                                    dec,
                                    source_cat=sourcecats[freq_label],
                                )
                                snr = get_snr(rho_mf, kappa_mf)
                                enmap.write_map(
                                    op.join(
                                        mfdir,
                                        f"{array_label}_{freq_label}_{int(ctime_)}_mf.fits",
                                    ),
                                    snr,
                                )
            if mpi:
                comm.barrier()

    # convert to dataframe
    if not mpi:
        light_curve = pd.DataFrame.from_dict(light_curve)
    else:
        light_curve = comm.gather(light_curve, root=0)
        if rank == 0:
            assert len(light_curve) == size
            light_curve = [lc for lc in light_curve if len(lc["ctime"]) > 0]
            light_curve = pd.concat([pd.DataFrame.from_dict(lc) for lc in light_curve])

            # sort by freq, array, ctime
            light_curve = light_curve.sort_values(by=["freq", "array", "ctime"])

    if rank == 0:
        # add units
        light_curve["ra"].unit = u.rad
        light_curve["dec"].unit = u.rad
        light_curve["flux"].unit = u.mJy
        light_curve["dflux"].unit = u.mJy
        light_curve["ctime"].unit = u.s
        if pol:
            light_curve["Q"].unit = u.mJy
            light_curve["U"].unit = u.mJy
            light_curve["dQ"].unit = u.mJy
            light_curve["dU"].unit = u.mJy

        return rank, light_curve
    else:
        return rank, None


def plot_lightcurve(data, oname, ctime=None, ylog=False):
    freq = ["f090", "f150", "f220", "f030", "f040"]
    colorcode = ["k", "b", "g", "r", "c"]

    plt.figure()
    for i, f in enumerate(freq):
        data_f = data[data["freq"] == f]
        plt.errorbar(
            data_f["ctime"],
            data_f["flux"],
            yerr=data_f["dflux"],
            fmt=f"{colorcode[i]}.",
            label=f,
        )

    # plot dotted vertical line at ctime
    for c in ctime:
        plt.axvline(x=c, color="r", linestyle="--")

    if ylog:
        plt.yscale("log")
    plt.xlabel("ctime")
    plt.ylabel("flux [mJy]")
    plt.legend()
    plt.savefig(oname)

    plt.close()

    return


def run_forced_photometry(
    ra_deg, dec_deg, ctime_start, ctime_end, pmat_psrc_rsigma, highpass, outdir
):
    """
    Make light curve in forced photometry way

    Args:
        ra_deg: ra in degree
        dec_deg: dec in degree
        ctime_start: start in ctime
        ctime_end: end in ctime
        pmat_psrc_rsigma:
        highpass:parameter for highpass filter, default=1
        outdir: in format like ~/000x/000x_ind_pax_fxxx/ or xxx/000x/000x_ind_general/
    Returns:
        hdf file constaining each detectors response at ra and dec saved in desired directory in hdf format
    """
    cat_file = outdir + "/cat.txt"
    cat = np.array([ra_deg, dec_deg, 1, 1])
    cat = cat.reshape((1, cat.shape[0]))
    np.savetxt(cat_file, cat)
    os.system(
        'python /scratch/gpfs/snaess/actpol/planet9/20200801/follow_up/20201030/forced_photometry_subarray.py %s "t>=%s,t<%s" %s --pmat_ptsrc_rsigma=%s --highpass=%s'
        % (cat_file, ctime_start, ctime_end, outdir, pmat_psrc_rsigma, highpass)
    )


def run_forced_photometry_sbatch(
    cat_repeat, group, period, ocat_path, slurmpath, outdir, submit
):
    """
    Make and submit slurm to do forced photometry
    Args:
        cat_repeat : dataframe read from catalog_repeat.csv
        group : the group number
        period : plus and minus days
        ocat_path : crossmatch output ocat path
        slurmpath : path to save the slurm file
    Returns:
        make hdf file containing each detectors response at ra and dec saved in desired directory in hdf format
    """
    order = 0
    for i in range(cat_repeat.shape[0]):
        if cat_repeat.group[i] == group:
            repeat = cat_repeat.n_repeats[i]
            ra_gen = cat_repeat.ra[i]
            dec_gen = cat_repeat.dec[i]
            name = cat_repeat.id[i]
            ctime_gen = float(name.split("_")[0])
            opath_new = outdir + "/%s/%s_%d_general/" % (
                str(group).zfill(4),
                str(group).zfill(4),
                order,
            )
            if not os.path.exists(opath_new):
                os.makedirs(opath_new)
            cat_file = opath_new + "/cat.txt"
            cat = np.array([[ra_gen, dec_gen, 1, 1]])
            np.savetxt(cat_file, cat)
            slurmfile = slurmpath + "/%s_%d_general.slurm" % (
                str(group).zfill(4),
                order,
            )
            with open(slurmfile, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(
                    """#SBATCH --nodes %d --ntasks-per-node=%d --cpus-per-task=%d --time=%s --job-name %s\n"""
                    % (1, 1, 20, "20:00:00", "map")
                )
                f.write(
                    'OMP_NUM_THREADS=20 python /scratch/gpfs/snaess/actpol/planet9/20200801/follow_up/20201030/forced_photometry_subarray.py %s "t>=%s,t<%s" %s --pmat_ptsrc_rsigma=3 --highpass=1'
                    % (
                        cat_file,
                        ctime_gen - period * 24 * 3600,
                        ctime_gen + period * 24 * 3600,
                        opath_new,
                    )
                )
                f.close()
            if submit == True:
                os.system("sbatch %s" % slurmfile)
            ocatpath = ocat_path + "/%s/%s_ocat.fits" % (name, name)
            ocat = fits.open(ocatpath)[1].data
            for j in range(ocat.shape[0]):
                band = ocat.band[j]
                opath_new = outdir + "/%s/%s_%d_%s/" % (
                    str(group).zfill(4),
                    str(group).zfill(4),
                    order,
                    band,
                )
                if not os.path.exists(opath_new):
                    os.makedirs(opath_new)
                ctime = float(ocat.ctime[j])
                ra = ocat.ra[j]
                dec = ocat.dec[j]
                cat_file = opath_new + "/cat.txt"
                cat = np.array([[ra, dec, 1, 1]])
                np.savetxt(cat_file, cat)
                slurmfile = slurmpath + "/%s_%d_%s.slurm" % (
                    str(group).zfill(4),
                    order,
                    band,
                )
                with open(slurmfile, "w") as f:
                    f.write("#!/bin/bash\n")
                    f.write(
                        """#SBATCH --nodes %d --ntasks-per-node=%d --cpus-per-task=%d --time=%s --job-name %s\n"""
                        % (1, 1, 20, "20:00:00", "map")
                    )
                    f.write(
                        'OMP_NUM_THREADS=20 python /scratch/gpfs/snaess/actpol/planet9/20200801/follow_up/20201030/forced_photometry_subarray.py %s "t>=%s,t<%s" %s --pmat_ptsrc_rsigma=3 --highpass=1'
                        % (
                            cat_file,
                            ctime_gen - period * 24 * 3600,
                            ctime_gen + period * 24 * 3600,
                            opath_new,
                        )
                    )
                    f.close()
                if submit == True:
                    os.system("sbatch %s" % slurmfile)
            order += 1


def run_lightcurve(path, div):
    hdf_files = gl.glob(path + "/*.hdf")
    ctimes = []
    for i in hdf_files:
        filename = i.split("/")[-1]
        print(filename)
        hdf_ctimes = filename.split("_")[1]
        hdf_ctime = hdf_ctimes.split(".")[0]
        ctimes.append(int(hdf_ctime))
    ctimes_set = set(ctimes)
    ctimes_list = list(ctimes_set)
    print(ctimes_list)

    def cluster(data, maxgap):
        data.sort()
        groups = [[data[0]]]
        for x in data[1:]:
            if abs(x - groups[-1][-1]) <= maxgap:
                groups[-1].append(x)
            else:
                groups.append([x])
        return groups

    ctime_groups = cluster(ctimes_list, 700)
    for group in ctime_groups:
        ctime1_group = str(group[0])
        ctimes_str = ""
        for ctime in group:
            ctime_str = path + "/flux_%s.*.hdf " % ctime
            ctimes_str += ctime_str
            print(ctimes_str)
        try:
            filename = path + "result_%d_%s.txt" % (div, ctime1_group)
            os.system(
                "python /scratch/gpfs/snaess/actpol/planet9/20200801/follow_up/20201030/subarray_lightcurve.py %s %s %s -n %d "
                % (0, ctimes_str, filename, div)
            )
        except:
            print("file empty")
