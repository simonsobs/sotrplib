"""
Emily Biermann
6/30/23

Script to test transient pipeline driver function

Updated Oct 2024, by Allen Foster 
Testing for use as SO Time domain pipeline test.

"""

import os.path as op
from sotdplib.pipeline import depth1_pipeline, Depth1
from pixell import enmap
import argparse
from astropy.table import Table

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--map_path",
    help="path to maps",
    default="/scratch/gpfs/snaess/actpol/maps/depth1/release/15064/depth1_1506466826_pa4_f150_map.fits",
)
parser.add_argument("-g", "--gridsize", help="gridsize in deg", default=1.0)
parser.add_argument(
    "--solarsystem",
    help="path asteroid ephem directories",
    default="/scratch/gpfs/snaess/actpol/ephemerides/objects",
)
parser.add_argument(
    "--plot_output",
    help="path to output maps and code for plots",
    default="/scratch/gpfs/amfoster/so/test_transient_pipeline/",
)

args = parser.parse_args()

sourcecat_fname = "/home/eb8912/transients/depth1/ACT_depth1-transient-pipeline/data/inputs/PS_S19_f090_2pass_optimalCatalog.fits"
galmask_fname = "/home/eb8912/transients/depth1/ACT_depth1-transient-pipeline/data/inputs/mask_for_sources2019_plus_dust.fits"
sourcecat = Table.read(sourcecat_fname)
galmask = enmap.read_map(galmask_fname)
plotdir = op.join(args.plot_output, args.map_path.split("/")[-1][:-9])

path = args.map_path[:-8]
imap = enmap.read_map(args.map_path, sel=0)
ivar = enmap.read_map(path + "ivar.fits")
rho = enmap.read_map(path + "rho.fits", sel=0)
kappa = enmap.read_map(path + "kappa.fits", sel=0)
time = enmap.read_map(path + "time.fits")
arr = args.map_path.split("/")[-1].split("_")[2]
freq = args.map_path.split("/")[-1].split("_")[3]
ctime = float(args.map_path.split("/")[-1].split("_")[1])
data = Depth1(imap, ivar, rho, kappa, time, arr, freq, ctime)


# test Pipeline
ocat = depth1_pipeline(
    data,
    args.gridsize,
    10,
    galmask,
    sourcemask=sourcecat,
    ast_dir=args.solarsystem,
    verbosity=2,
)
print(ocat)


for source in ocat:
    print(source)
# all_sources = tools.get_ps_inmap(imap, sourcecat, inputs.get_sourceflux_threshold(freq))
# print(all_sources)
