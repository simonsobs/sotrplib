from datetime import datetime, timezone
from glob import glob

import numpy as np
from astropy import units as u
from astropy.io import fits
from pixell import enmap
from tqdm import tqdm

from sotrplib.maps.core import IntensityAndInverseVarianceMap
from sotrplib.maps.map_coadding import RhoKappaMapCoadder
from sotrplib.maps.preprocessor import (
    EdgeMask,
    KappaRhoCleaner,
    MatchedFilter,
    PlanetMasker,
)

pm = PlanetMasker()
mf = MatchedFilter(
    band_height=1 * u.deg,
    shrink_holes=5 * u.arcmin,
    noisemask_lim=0.03,
    apod_holes=10 * u.arcmin,
)
kc = KappaRhoCleaner()
iv_em = EdgeMask(mask_on="inverse_variance")
em = EdgeMask(mask_on="kappa", edge_width=10 * u.arcmin)


band = "f090"
ivar_files = glob(
    f"/scratch/gpfs/SIMONSOBS/users/amfoster/so/lat_early_maps/out_deep56/17*/depth1_*_{band}_ivar.fits"
)
box = None

coadder = RhoKappaMapCoadder(frequencies=[band])


def get_obs_times(timefile):
    with fits.open(timefile) as hdul:
        t = hdul[0].data
        valid_times = t > 0.0
    t0 = float(timefile.split("/")[-1].split("_")[1])

    return t0 + np.amin(t[valid_times]), t0 + np.amax(t[valid_times])


map_coadds = []
for i in tqdm(range(len(ivar_files))):
    start, stop = get_obs_times(ivar_files[i].split("_ivar.fits")[0] + "_time.fits")
    imap = IntensityAndInverseVarianceMap(
        intensity_filename=ivar_files[i].split("ivar.fits")[0] + "map.fits",
        inverse_variance_filename=ivar_files[i],
        time_filename=ivar_files[i].split("_ivar.fits")[0] + "_time.fits",
        info_filename=ivar_files[i].split("_ivar.fits")[0] + "_info.hdf5",
        start_time=datetime.fromtimestamp(int(start), tz=timezone.utc),
        end_time=datetime.fromtimestamp(int(stop), tz=timezone.utc),
        frequency=band,
        array=ivar_files[i].split("_")[-3],
        intensity_units=u.uK,
        box=box,
    )

    imap.build()

    filtered_map = em.preprocess(kc.preprocess(mf.preprocess(pm.preprocess(imap))))

    map_coadds = coadder.coadd(map_coadds + [filtered_map])

for mc in map_coadds:
    enmap.write_map(
        f"/scratch/gpfs/SIMONSOBS/users/amfoster/so/lat_early_maps/out_deep56/coadds/{band}_coadded_rho.fits",
        mc.rho,
    )
    enmap.write_map(
        f"/scratch/gpfs/SIMONSOBS/users/amfoster/so/lat_early_maps/out_deep56/coadds/{band}_coadded_kappa.fits",
        mc.kappa,
    )

    mc.finalize()
    enmap.write_map(
        f"/scratch/gpfs/SIMONSOBS/users/amfoster/so/lat_early_maps/out_deep56/coadds/{band}_coadded_flux.fits",
        mc.flux,
    )
    enmap.write_map(
        f"/scratch/gpfs/SIMONSOBS/users/amfoster/so/lat_early_maps/out_deep56/coadds/{band}_coadded_snr.fits",
        mc.snr,
    )
    enmap.write_map(
        f"/scratch/gpfs/SIMONSOBS/users/amfoster/so/lat_early_maps/out_deep56/coadds/{band}_coadded_time_mean.fits",
        mc.time_mean,
    )
    enmap.write_map(
        f"/scratch/gpfs/SIMONSOBS/users/amfoster/so/lat_early_maps/out_deep56/coadds/{band}_coadded_time_first.fits",
        mc.time_first,
    )
    enmap.write_map(
        f"/scratch/gpfs/SIMONSOBS/users/amfoster/so/lat_early_maps/out_deep56/coadds/{band}_coadded_time_last.fits",
        mc.time_last,
    )
    enmap.write_map(
        f"/scratch/gpfs/SIMONSOBS/users/amfoster/so/lat_early_maps/out_deep56/coadds/{band}_coadded_hits.fits",
        mc.hits,
    )
