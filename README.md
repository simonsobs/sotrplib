# sotrplib
Simons Observatory Time Resolved Pipeline Library


Porting files from the ACT depth-1 transient pipeline. 
Creating a test setup of the SO map-domain transient pipeline which will start by ingesting .FITS depth-1 maps.


To run the pipeline, you can simply do:

python trp_pipeline.py --maps  <path/to/map.fits> --save-json

for example:
python trp_pipeline.py --maps  /scratch/gpfs/SIMONSOBS/so/maps/actpol/depth1/15001/depth1_1500110453_pa5_f090_rho.fits /scratch/gpfs/SIMONSOBS/users/amfoster/scratch/depth1_month_coadds/coadd_1516177324.0_all_f090_rho.fits --save-json

will run the msaking, cleaning and source finding on the two input maps, match the output to the act source catalog and then decide which are transients or noise based on simple cuts.

can also provide several depth-1 maps as input and coadd over n days by supplying the --coadd-n-days argument.
