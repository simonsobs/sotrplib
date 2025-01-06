from glob import glob

import argparse as ap
P = ap.ArgumentParser(description="Collate single depth-1 map flux files into a single lightcurve file.",
                      formatter_class=ap.ArgumentDefaultsHelpFormatter,
                     )

P.add_argument("--files",
               action="store",
               default=[],
               nargs='+',
               help="Files to collate. if left empty and --dir supplied, then glob that dir "
              )

P.add_argument("--lc-dir",
               action="store",
               default='./',
               help="Directory containing the single depth-1 lightcurve files to be collated."
              )

P.add_argument("--out-dir",
               action="store",
               default='./',
               help="Directory to store output file."
              )

P.add_argument("--out-file",
               action="store",
               default='collated_lightcurve.txt',
               help="Output file name."
              )

P.add_argument("--default-file-suffix",
               action="store",
               default='tmp_lightcurve.txt',
               help="Suffix of temporary lightcurve files (so that deletion is safer)."
              )
P.add_argument("--default-thumbnail-file-suffix",
               action="store",
               default='tmp_thumbnail.hdf5',
               help="Suffix of temporary thumbnail files (so that deletion is safer)."
              )
## not used yet...
P.add_argument("--split-sources",
               action="store_true",
               default=False,
               help="Split final lightcurve file by source, current uses ra,dec but can eventually use sourceID, say for asteroids."
              )
P.add_argument("--no-thumbnails",
               action="store_true",
               default=False,
               help="Ignore the thumbnail files."
              )
P.add_argument("--no-cleanup",
               action="store_true",
               default=False,
               help="Dont cleanup the initial lc_dir."
              )

def collate_lightcurve_files(lc_files,
                             out_file,
                             cleanup=True
                            ):
    from tqdm import tqdm
    inlines = []
    for lcf in tqdm(sorted(lc_files),desc='Reading lightcurve files'):
        with open(lcf,'r') as f:
            for line in f:
                inlines.append(line)

    with open(out_file,'w') as f:
        for i in range(len(inlines)):
            f.write(inlines[i])
        
    if cleanup:
        from subprocess import run as sprun
        for lcf in tqdm(sorted(lc_files),desc='Cleaning up lightcurve files'):
            sprun(['rm',lcf])
    return

def load_custom_hdf5(fname):
    import h5py
    from pixell import enmap
    context = h5py.File(fname, "r")
    thumbnails = []
    for key in context:
        thumb_dict = {}
        thumb_dict['thumb'] = enmap.read_hdf(context[key])
        for k in context[key].keys():
            if k == 'wcs' or k =='data':
                continue
            thumb_dict[k] = enmap.fix_python3(context[key][k][()])
        
        thumbnails.append(thumb_dict)
    return thumbnails

def collate_thumbnail_files(files,
                            out_file,
                            cleanup=True
                           ):
    from pixell import enmap
    from tqdm import tqdm
    thumbnails = []
    for f in tqdm(sorted(files),desc='Loading thumbnail files...'):
       thumbnails += load_custom_hdf5(f)

    for i in tqdm(range(len(thumbnails)),desc='Writing thumbnails to file'):
        t = thumbnails[i].pop('thumb')
        enmap.write_hdf(out_file, 
                        t,
                        address=str(i).zfill(len(str(len(thumbnails)))),
                        extra=thumbnails[i]
                       )
        
    if cleanup:
        from subprocess import run as sprun
        for f in tqdm(sorted(files),desc='Cleaning up thumbnails'):
            sprun(['rm',f])
    return

args = P.parse_args()

lc_files = args.files
if len(args.files)==0:
    lc_files = glob(args.lc_dir+f'*{args.default_file_suffix}')

collate_lightcurve_files(lc_files,
                         args.out_dir+args.out_file,
                         cleanup=not bool(args.no_cleanup)
                         )

thumb_files = glob(args.lc_dir+f'*{args.default_thumbnail_file_suffix}')
if 'lightcurve' in args.out_file:
    outfile = args.out_dir+args.out_file.split('lightcurve.txt')[0]+'thumbnail.hdf5'
else:
    outfile = args.out_dir+args.out_file.split('.txt')[0]+'.hdf5'

collate_thumbnail_files(thumb_files,
                        outfile,
                        cleanup=not bool(args.no_cleanup)
                       )