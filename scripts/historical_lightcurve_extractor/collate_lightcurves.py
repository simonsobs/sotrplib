import argparse as ap
from glob import glob

P = ap.ArgumentParser(
    description="Collate single depth-1 map flux files into a single lightcurve file.",
    formatter_class=ap.ArgumentDefaultsHelpFormatter,
)

P.add_argument(
    "--files",
    action="store",
    default=[],
    nargs="+",
    help="Files to collate. if left empty and --dir supplied, then glob that dir ",
)

P.add_argument(
    "--lc-dir",
    action="store",
    default="./",
    help="Directory containing the single depth-1 lightcurve files to be collated.",
)

P.add_argument(
    "--out-dir", action="store", default="./", help="Directory to store output file."
)

P.add_argument(
    "--out-file",
    action="store",
    default="collated_lightcurve.txt",
    help="Output file name.",
)

P.add_argument(
    "--default-file-suffix",
    action="store",
    default="tmp_lightcurve.txt",
    help="Suffix of temporary lightcurve files (so that deletion is safer).",
)
P.add_argument(
    "--default-thumbnail-file-suffix",
    action="store",
    default="tmp_thumbnail.hdf5",
    help="Suffix of temporary thumbnail files (so that deletion is safer).",
)
## implement splitting thumbnails by source. each source has different file.
P.add_argument(
    "--split-sources",
    action="store_true",
    default=False,
    help="Split final lightcurve file by source, current uses ra,dec but can eventually use sourceID, say for asteroids.",
)
P.add_argument(
    "--no-thumbnails",
    action="store_true",
    default=False,
    help="Ignore the thumbnail files.",
)
P.add_argument(
    "--no-cleanup",
    action="store_true",
    default=False,
    help="Dont cleanup the initial lc_dir.",
)
P.add_argument(
    "--overwrite",
    action="store_true",
    default=False,
    help="Overwrite output files if they exist.",
)


def collate_lightcurve_files(lc_files, out_file, cleanup=True):
    from tqdm import tqdm

    inlines = []
    for lcf in tqdm(sorted(lc_files), desc="Reading lightcurve files"):
        with open(lcf, "r") as f:
            for line in f:
                inlines.append(line)

    headerlines = 0
    with open(out_file, "a" if not args.overwrite else "w") as f:
        for i in range(len(inlines)):
            if "#" in inlines[i]:
                headerlines += 1
                if headerlines > 1:
                    continue
            f.write(inlines[i])

    if cleanup:
        from subprocess import run as sprun

        for lcf in tqdm(sorted(lc_files), desc="Cleaning up lightcurve files"):
            sprun(["rm", lcf])
    return


def load_custom_hdf5(fname):
    import h5py
    from pixell import enmap

    try:
        context = h5py.File(fname, "r")
    except Exception as e:
        print(e, fname)
        return []
    thumbnails = []
    for key in context:
        thumb_dict = {}
        try:
            thumb_dict["thumb"] = enmap.read_hdf(context[key])
        except Exception as e:
            print(e, key, context[key])
            continue
        for k in context[key].keys():
            if k == "wcs" or k == "data":
                continue
            thumb_dict[k] = enmap.fix_python3(context[key][k][()])

        thumbnails.append(thumb_dict)
    return thumbnails


def collate_thumbnail_files(files, out_file, split_sources=False, cleanup=True):
    from pixell import enmap
    from tqdm import tqdm

    old_i = 0
    for f in tqdm(sorted(files), desc="Loading thumbnail files..."):
        thumbnails = []
        thumbnails += load_custom_hdf5(f)
        len_objs = len(thumbnails) + old_i
        for i in range(len(thumbnails)):
            if "thumb" in thumbnails[i]:
                t = thumbnails[i].pop("thumb")
                if split_sources:
                    ## source name is SO-S Jxxxxx-xxxx so , just remove the SO-S part
                    source_name = thumbnails[i].get("name", "???").split(" ")[1]
                    output_file = f"{out_file.split('.hdf5')[0]}_{source_name}.hdf5"
                else:
                    output_file = out_file

                try:
                    enmap.write_hdf(
                        output_file,
                        t,
                        address=str(old_i + i).zfill(len(str(len_objs))),
                        extra=thumbnails[i],
                    )
                except Exception as e:
                    print(e, i, thumbnails[i], str(old_i + i).zfill(len(str(len_objs))))
                    continue
        old_i = len_objs + 1

        if cleanup:
            from subprocess import run as sprun

            sprun(["rm", f])
    return


args = P.parse_args()

lc_files = args.files
if len(args.files) == 0:
    lc_files = glob(args.lc_dir + f"*{args.default_file_suffix}")

collate_lightcurve_files(
    lc_files, args.out_dir + args.out_file, cleanup=not bool(args.no_cleanup)
)

thumb_files = glob(args.lc_dir + f"*{args.default_thumbnail_file_suffix}")
if "lightcurve" in args.out_file:
    outfile = args.out_dir + args.out_file.split("lightcurve.txt")[0] + "thumbnail.hdf5"
else:
    outfile = args.out_dir + args.out_file.split(".txt")[0] + ".hdf5"

collate_thumbnail_files(
    thumb_files,
    outfile,
    split_sources=args.split_sources,
    cleanup=not bool(args.no_cleanup),
)
