## initially from Sigurd Naess, downloaded from https://phy-act1.princeton.edu/~snaess/actpol/lightcurves/20220624/depth1_lightcurves.py

import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("--ra",  default=[],type=float,nargs='+')
parser.add_argument("--dec", default=[],type=float,nargs='+')

parser.add_argument("--rho-maps", nargs="+", default=[])
parser.add_argument("--odir",default='./lightcurves/')
parser.add_argument("--scratch-dir",default='./lightcurves/tmp/')
parser.add_argument("--ofname",default='tmp_lightcurves.txt')
parser.add_argument("--coadd-dir",default='/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/coadds/')
parser.add_argument("--subtract-coadd",action='store_true')
parser.add_argument("-s", "--snmin", type=float, default=None)
parser.add_argument("-n", "--nmax",  type=int,   default=None)
parser.add_argument("-T", "--tol",   type=float, default=1e-4)
parser.add_argument("-S", "--fitlim",type=float, default=0)
parser.add_argument("-c", "--cont",  action="store_true")

args = parser.parse_args()

import numpy as np
from pixell import enmap, utils, bunch, mpi
import os
from glob import glob

# 1. Single file with all data:
#    ctime src arr ftag snr T dT Q dQ U dU
#    +: Simple, easy to move around
#    -: gnuplot-unfriendly, hard to add more srcs, need postproc
# 2. One file per src
#    +: better for gnuplot, easy to add more srcs
#    -: need postproc
# 3. Structured file, with all array-freqs for a given depth1-tag on
#    the same line
#    +: Good for things like spectral index calculation
#    -: Will have lots of empty entries
#
# I'll go with #1 for now. Can always reformat to something else later

def get_time_safe(time_map, poss, r=5*utils.arcmin):
    # First try to read off directly
    poss     = np.array(poss)
    vals     = time_map.at(poss, order=0)
    bad      = np.where(vals==0)[0]
    if len(bad) > 0:
        # This shouldn't be too slow as long as the number of sources isn't too big
        pixboxes = enmap.neighborhood_pixboxes(time_map.shape, time_map.wcs, poss.T[bad], r=r)
        for i, pixbox in enumerate(pixboxes):
            thumb = time_map.extract_pixbox(pixbox)
            mask  = thumb != 0
            vals[bad[i]] = np.sum(mask*thumb)/np.sum(mask)
    return vals

def fit_poss(rho, kappa, poss, rmax=8*utils.arcmin, tol=1e-4, snmin=3):
    """Given a set of fiducial src positions [{dec,ra},nsrc],
    return a new set of positions measured from the local center-of-mass
    Assumes scalar rho and kappa"""
    from scipy import ndimage
    ref     = np.max(kappa)
    if ref == 0: ref = 1
    snmap2  = rho**2/np.maximum(kappa, ref*tol)
    # label regions that are strong enough and close enough to the
    # fiducial positions
    mask    = snmap2 > snmin**2
    mask   &= snmap2.distance_from(poss, rmax=rmax) < rmax
    labels  = enmap.samewcs(ndimage.label(mask)[0], rho)
    del mask
    # Figure out which labels correspond to which objects
    label_inds = labels.at(poss, order=0)
    good       = label_inds > 0
    # Compute the center-of mass position for the good labels
    # For the bad ones, just return the original values
    oposs = poss.copy()
    if np.sum(good) > 0:
        oposs[:,good] = snmap2.pix2sky(np.array(ndimage.center_of_mass(snmap2, labels, label_inds[good])).T)
    del labels
    osns  = snmap2.at(oposs, order=1)**0.5
    #for i in range(nsrc):
    #dpos = utils.rewind(oposs[:,i]-poss[:,i])
    #print("%3d %6.2f %8.3f %8.3f %8.3f %8.3f" % (i, osns[i], poss[1,i]/utils.degree, poss[0,i]/utils.degree, dpos[1]/utils.arcmin, dpos[0]/utils.arcmin))
    return oposs, osns

comm     = mpi.COMM_WORLD
# Get our input map files
rhofiles = sum([sorted(glob(fname)) for fname in args.rho_maps],[])
nfile    = len(rhofiles)

## get ra,dec into format used by pixell sky2pix
poss = np.array([[d*utils.degree for d in args.dec],[r*utils.degree for r in args.ra]])

# Process our files. We'll make a separate lightcurve file per map, and then
# merge them in the end
lines = []

for fi in range(comm.rank, nfile, comm.size):
    rhofile   = rhofiles[fi]
    kappafile = utils.replace(rhofile, "rho", "kappa")
    timefile  = utils.replace(rhofile, "rho", "time")
    infofile  = utils.replace(utils.replace(rhofile, "rho", "info"), ".fits", ".hdf")
    name      = utils.replace(os.path.basename(rhofile), "_rho.fits", "")
    ttag, arr, ftag = name.split("_")[1:4]
    
    if args.subtract_coadd:
        coadd = args.coadd_dir+f'act_daynight_{ftag}_map.fits'
    else:
        coadd = None
    
    # Check if any sources are inside our geometry
    shape, wcs = enmap.read_map_geometry(rhofile)
    pixs       = enmap.sky2pix(shape, wcs, poss)
    inside     = np.where(np.all((pixs.T >= 0)&(pixs.T<shape[-2:]),-1))[0]
    print(inside)
    print("Processing %s with %4d srcs" % ( name, len(inside)))
    if len(inside) == 0:
        # Just create an empty file if we don't have any sources in this map
       continue
    else:
        ra = np.asarray(args.ra)[inside]
        dec = np.asarray(args.dec)[inside]
        # Otherwise process the map properly.
        # We read in and get values from one map at a time to save memory
        kappa_map = enmap.read_map(kappafile)
        # reference kappa value. Will be used to determine if individual kappa values are too low
        ref     = np.max(kappa_map)
        kappa   = kappa_map.at(poss[:,inside])
        if ref == 0: ref = 1
        rho_map = enmap.read_map(rhofile)
        if args.fitlim > 0:
            pos_fit, sn_fit = fit_poss(rho_map.preflat[0], kappa_map.preflat[0], poss[:,inside], tol=args.tol)
            good = sn_fit >= args.fitlim
            poss[:,inside[good]] = pos_fit[:,good]
        rho     = rho_map.at(poss[:,inside])
        del kappa_map, rho_map

        time_map  = enmap.read_map(timefile)
        info      = bunch.read(infofile)
        with utils.nowarn():
            t = get_time_safe(time_map, poss[:,inside])+info.t
        del time_map, info

        good   = np.where(kappa[0] > ref*args.tol)[0]
        print(good)
        rho, kappa, t = rho[:,good], kappa[:,good], t[good]

        if coadd:
            coadd_flux = enmap.read_map(coadd,sel=0).at(poss[:,inside])
        else:
            coadd_flux = 0.0

        flux   = rho/kappa - coadd_flux
        dflux  = kappa**-0.5
        ## intensity snr only
        snr    = flux[0]/dflux[0]
        if len(snr)==0:
            print(flux,dflux)
            continue
        
        for i, gi in enumerate(good):
            print(i,gi)
            print(flux[0,i],dflux[0,i])
            line = "%10.0f, %5f, %.5f, %3s, %4s, %8.2f," % (t[i], ra[gi], dec[gi], arr, ftag, snr[i])
            for f, df in zip(flux[:,i], dflux[:,i]):
                line += " %8.1f, %6.1f," % (f, df)
            line += " %s\n" % ttag
            lines.append(line)
        
ofile = "%s/%s" %(args.odir,args.ofname)
times = [float(line.split(',')[0]) for line in lines]
order = np.argsort(times)
with open(ofile, "w+") as f:
    for oi in order:
        f.write(lines[oi])


