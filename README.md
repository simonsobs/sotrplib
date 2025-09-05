# sotrplib
Simons Observatory Time Resolved Pipeline Library

Creating a test setup of the SO map-domain transient pipeline which will start by ingesting .FITS depth-1 maps.

Currently, the output is three pandas databases in csv format.

Now supports sim mode which can create gaussian noise maps, inject and recover sources and also inject sources into depth1 maps.
More below  in the ## simulations section

## Development requirements

To get ready for development, create a virtual enviroment and install the package:
```
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```
We use `ruff` for formatting. When you go to commit your code, it will automatically be 
formatted thanks to the pre-commit hook.

Tests are performed using `pytest`.

## extra required packages

astroquery
photutils

## SO depth-1 sims 
pipe-s0004 sims for 3 days and 4 optics tubes at 90 / 150 GHz are available in the productdb and have been saved to tiger3 in 
`/scratch/gpfs/SIMSONSOBS/users/amfoster/scratch/pipe-s0004_depth1/`

I've run the maps through the matched filtering function `matched_filter_depth1_map` in `sotrplib/filters/filters.py` using a beam fwhm of 2.2 arcmin at f090 and 1.4 arcmin at f150.

## Running the pipeline
An example use of the source extraction pipeline can be run using the script `sotrplib/trp_pipeline.py`

This script can also be used to coadd depth1 maps before source-finding.

Right now, sources are saved as pandas databases as `MeasuredSource` objects which will in the future be directly stored in the SO source catalog and lightcurve databases.

> *You have to create the output directory before running.*

### running on sims

Here is an example of running the source extraction pipeline on individual depth1 maps from the sims, using the input websky catalog as the known-source catalog.
Sources are extracted above s/n ratio of 5 and the input source catalog is limited to sources above 30mJy.
The `--simulation` tag tells the script to record the injected source fluxes given in the catalog along with the recovered source fluxes.
`--output-dir` tells the script where to save the files (also tells the script where to save the plots if you choose to make them.)

```py
python trp_pipeline.py --maps /scratch/gpfs/SIMONSOBS/users/amfoster/scratch/pipe-s0004_depth1/1696*/depth1*rho.fits --output-dir /scratch/gpfs/SIMONSOBS/users/[youruser]/scratch/pipe-s0004_depth1_extracted_sources/ -s 5 --verbose --source-catalog /scratch/gpfs/SIMONSOBS/users/amfoster/so/pipe-s0004_depth1_sims/websky_cat_100_1mJy.csv --flux-threshold 0.03 --simulation
```

> *Warning: the websky sims have a lot of sources, so make sure to set the flux threshold to avoid including the several hundred thousand low-flux sources.*

### Running on ACT depth1 maps

```py
python trp_pipeline.py --maps /scratch/gpfs/SIMONSOBS/so/maps/actpol/depth1/15873/*rho.fits --output-dir /scratch/gpfs/SIMONSOBS/users/[youruser]/scratch/act_depth1_extracted_sources/ -s 5 --verbose --source-catalog /scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/PS_S19_f090_2pass_optimalCatalog.fits
```

### Coadding maps

```py
python trp_pipeline.py --maps /scratch/gpfs/SIMONSOBS/so/maps/actpol/depth1/158*/*rho.fits --output-dir /scratch/gpfs/SIMONSOBS/users/[youruser]/scratch/act_depth1_extracted_sources/ -s 5 --verbose --source-catalog /scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/PS_S19_f090_2pass_optimalCatalog.fits --coadd-n-days 7
```

This script will run the same way as the previous example, however it will coadd observations in 7 day bins before performing the source-finding. 

## Basic operation

The trp_pipeline consists of the following steps:

- Group maps (i.e. Figure out which maps belong in which time bin)
- Load map or coadd maps within one time bin.
- Perform preprocessing (i.e. `clean_map` and `kappa_clean`, masking, and perhaps flatfielding if depth1)
- Use 2D gaussian fit on known bright sources, calculates offsets
- Forced photometry on other known sources
- Mask known sources
- Blind search for sources above user-input SNR threshold.
- Compare extracted sources to external catalog(s); i.e. perform the sift operation.
- Output the source candidates, transient candidates, and noise candidates to pandas database, sifted via the sifter


## Simulations

The trp_pipeline supports three modes of source / transient injection:

1. Gaussian white noise map with injected sources. User inputs map / source properties via `sims/sim_config.yaml`
2. Inject sources directly into the input maps which are read in from disk. Uses the `sim_config.yaml`.
3. Use input maps to get the map extent and time map, then make gaussian noise realization and inject sources via `sim_config.yaml`

Running the `trp_pipeline.py` script with the `--sim` argument will perform number 1. The `--sim-config` argument should give the path to the sim config yaml file.
The `--simulated-transient-database` argument can be passed pointing to a premade database of transient sources (see for example `scripts/act_depth1_transient_sims/generate_transients_for_act.py`).
To inject transients from the config file, use `--inject-transients`, otherwise only the static sources will be injected.
If you want to inject transients directly into real maps (i.e. 2. from above) , you can use the `--inject-transients` argument without running `--sim`.
The final option, which reads in the real maps but then generates gaussian sims using that geometry and time map, one can run in `--sim` mode with `--inject-transients`  and `--use-map-geometry` .
