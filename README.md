# sotrplib
Simons Observatory Time Resolved Pipeline Library


Porting files from the ACT depth-1 transient pipeline. 
Creating a test setup of the SO map-domain transient pipeline which will start by ingesting .FITS depth-1 maps.


To run the pipeline, you can simply do:

`python trp_pipeline.py --plot-output <path/to/output/dir/>`

Which runs with a default map file (`/scratch/gpfs/snaess/actpol/maps/depth1/release/15064/depth1_1506466826_pa4_f150_map.fits`)

Otherwise, if you want to run multiple maps through, you can specify the input files using the `-m` argument, for example:

`python trp_pipeline.py -m obs1_map.fits obs2_map.fits obs3_map.fits --plot-output <path/to/output/dir/>`
