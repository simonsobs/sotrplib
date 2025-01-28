# sotrplib
Simons Observatory Time Resolved Pipeline Library


Porting files from the ACT depth-1 transient pipeline. 
Creating a test setup of the SO map-domain transient pipeline which will start by ingesting .FITS depth-1 maps.

## SO depth-1 sims 
pipe-s0004 sims for 3 days and 4 optics tubes at 90 / 150 GHz are available in the productdb and have been saved to tiger3 in 
`/scratch/gpfs/SIMSONSOBS/users/amfoster/scratch/pipe-s0004_depth1/`

I've run the maps through the matched filtering function `matched_filter_depth1_map` in `sotrplib/filters/filters.py` using a beam fwhm of 2.2 arcmin at f090 and 1.4 arcmin at f150.

## Running the pipeline
An example use of the source extraction pipeline can be run using the script `sotrplib/trp_pipeline.py`

This script can also be used to coadd depth1 maps before source-finding.

Right now, sources are saved in json files as `SourceCandidate` objects which will in the future be directly stored in the SO source catalog and lightcurve databases.

You have to create the output directory before running.

### running on sims

Here is an example of running the source extraction pipeline on individual depth1 maps from the sims, using the input websky catalog as the known-source catalog.
Sources are extracted above s/n ratio of 5 and the input source catalog is limited to sources above 30mJy.
The `--simulation` tag tells the script to record the injected source fluxes given in the catalog along with the recovered source fluxes.
`--save-json` tells the script to save the extracted sources in .json format and `--plot-output` tells the script where to save the files (name should be changed, but also tells the script where to save the plots if you choose to make them.)

```python trp_pipeline.py --maps /scratch/gpfs/SIMONSOBS/users/amfoster/scratch/pipe-s0004_depth1/1696*/depth1*rho.fits --save-json --plot-output /scratch/gpfs/SIMONSOBS/users/[youruser]/scratch/pipe-s0004_depth1_extracted_sources/ -s 5 --verbose --source-catalog /scratch/gpfs/SIMONSOBS/users/amfoster/so/pipe-s0004_depth1_sims/websky_cat_100_1mJy.csv --flux-threshold 0.03 --simulation```

Warning: the websky sims have a lot of sources, so make sure to set the flux threshold to avoid including the several hundred thousand low-flux sources.

### Running on ACT depth1 maps

```python trp_pipeline.py --maps /scratch/gpfs/SIMONSOBS/so/maps/actpol/depth1/15873/*rho.fits --save-json --plot-output /scratch/gpfs/SIMONSOBS/users/[youruser]/scratch/act_depth1_extracted_sources/ -s 5 --verbose --source-catalog /scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/PS_S19_f090_2pass_optimalCatalog.fits```

### Coadding maps

```python trp_pipeline.py --maps /scratch/gpfs/SIMONSOBS/so/maps/actpol/depth1/158*/*rho.fits --save-json --plot-output /scratch/gpfs/SIMONSOBS/users/[youruser]/scratch/act_depth1_extracted_sources/ -s 5 --verbose --source-catalog /scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/PS_S19_f090_2pass_optimalCatalog.fits --coadd-n-days 7 ```

This script will run the same way as the previous example, however it will coadd observations in 7 day bins before performing the source-finding. 

## Basic operation

The trp_pipeline consists of the following steps:

- Group maps (i.e. Figure out which maps belong in which time bin)
- Load map or coadd maps within one time bin.
- Perform preprocessing (i.e. `clean_map` and `kappa_clean`, masking, and perhaps flatfielding if depth1)
- Extract sources.
- Compare extracted sources to external catalog(s); i.e. perform the sift operation.
- Output the source candidates, tagged via the sifter