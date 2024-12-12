from glob import glob
import numpy as np

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
## not used yet...
P.add_argument("--split-sources",
               action="store_true",
               default=False,
               help="Split final lightcurve file by source, current uses ra,dec but can eventually use sourceID, say for asteroids."
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
    inlines = []
    for lcf in sorted(lc_files):
        with open(lcf,'r') as f:
            for line in f:
                inlines.append(line)

    with open(out_file,'w') as f:
        for i in range(len(inlines)):
            f.write(inlines[i])
        
    if cleanup:
        from subprocess import run as sprun
        for lcf in sorted(lc_files):
            sprun(['rm',lcf])
    return

args = P.parse_args()

lc_files = args.files
if len(args.files)==0:
    lc_files = glob(args.lc_dir+f'*{args.default_file_suffix}')

collate_lightcurve_files(lc_files,
                         args.out_dir+args.out_file,
                         cleanup=~args.no_cleanup
                         )
