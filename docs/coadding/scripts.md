Helper scripts for coadds
=========================

The scripts in `scripts/coadding/` help you make coadds. They are not part
of the installed package. To use a script, run it from a checkout of the
repository.

| Script | What it does |
|---|---|
| `submit_week_coadds.py` | Writes `sotrp-coadd` configs and SLURM jobs for a set of time windows. |
| `coadd_maps.py` | An example that makes a coadd with the library directly. |


`submit_week_coadds.py`
-----------------------

The script divides a time range into windows. For each window and each
frequency, it writes one `sotrp-coadd` config and one SLURM job script. Each
job runs `sotrp-coadd` on its config (see [Make coadds](make_coadds.md)).

### Procedure

1. Write the configs and the job scripts:

   ```
   python scripts/coadding/submit_week_coadds.py \
     --database-name /path/to/mapcat.sqlite \
     --depth-one-parent /path/to/depth1/<run> \
     --output-dir /path/to/weekly_coadds \
     --repo-dir /path/to/your/sotrplib/checkout \
     --socat-db-path /path/to/socat.db \
     --ephem-file-path '' \
     --time 08:00:00
   ```

2. Examine the configs in `<output-dir>/configs/`.
3. Run the same command again with `--submit`. The script then submits each
   job with `sbatch`.

Without `--submit`, the script only writes the files.

### What the script writes

The script writes each window into a subdirectory. The name of the
subdirectory is the UTC start date of the window:

```
<output-dir>/
  20250904/  f090_i1i3i4i6_1756951951_{rho,kappa,flux,snr,hits,time_mean}.fits  (+ f150, f220, f280)
  20250911/
  ...
  configs/   week00_f090.json ...
  slurm/     week00_f090.slurm, week00_f090.log ...
```

The FITS files appear when the jobs run.

Each config uses these settings:

- `time_binning: "left-bound"`. Thus, each depth-1 map is in one coadd only,
  and there are no gaps. For example, the 24 weekly coadds of the deep56 run
  contain all 674 depth-1 maps, and each map is in one coadd only.
- These preprocessors, in sequence: `planet_mask` (15 arcmin),
  `asteroid_mask` (`--asteroid-mask-radius`), `matched_filter` (with the
  beam from `--beam1d-template`, if the file exists), `kappa_rho` and
  `edge_mask` (on kappa, 10 arcmin).
- `mapcat_registration` with the `--coadd-type` and the name
  `{frequency}_{window start}_{window stop}` (unix times). The config and
  job files have the name `week<NN>_<frequency>`.

Each job script sets `MAPCAT_DATABASE_NAME`, `MAPCAT_DEPTH_ONE_PARENT` and
`MAPCAT_DEPTH_ONE_COADD_PARENT`. If SOCat is on (`--use-socat`, default), it
also sets `socat_client_client_type=db`, and `socat_model_database_name` from
`--socat-db-path`.

### Windows

Each window has a length of `--window-days` days (default 7). The first
window starts at `--start-time`. The windows are not calendar weeks.

If you do not set `--start-time` and `--end-time`, the script uses the first
and the last observation in the database.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--database-name` | (necessary) | The mapcat SQLite database. |
| `--depth-one-parent` | (necessary) | The root directory of the depth-1 maps. |
| `--output-dir` | (necessary) | The root directory of the coadd FITS files. |
| `--coadd-parent` | `--output-dir` | The value of `MAPCAT_DEPTH_ONE_COADD_PARENT`. It must contain `--output-dir`. |
| `--window-days` | 7 | The length of each window in days. |
| `--start-time`, `--end-time` | the database range | The time range (ISO 8601). |
| `--frequencies` | f090 f150 f220 f280 | The bands. The script writes one job for each band. |
| `--array` | all arrays | Use the maps of one array only. |
| `--repo-dir` | the checkout of the script | Each job goes to this checkout and activates its `.venv`. Use a checkout that you do not delete. |
| `--socat-db-path` | none | The SOCat database for the asteroid mask. |
| `--ephem-file-path` | a JPL file in the repository | The ephemeris file if SOCat is not available. Set `''` to use SOCat only. Then, if SOCat is not configured, the job fails. |
| `--no-register-coadds` | register | Write the FITS files, but do not register the coadds in mapcat. |
| `--rerun` | off | This option has an effect only if `track_processing` is true. See [Processing status](analyze_coadds.md#processing-status). |
| `--time` | 04:00:00 | The SLURM time limit. See "Job time" below. |
| `--submit` | off | Submit the jobs with `sbatch`. |

Run the script with `--help` for all options.

### Job time

Set `--time` for the number of maps in a window. Each input map takes
approximately 5 to 6 minutes. A busy week (approximately 50 maps at f090 or
f150) takes approximately 5 hours. **The default time limit of 4 hours is
too short for these weeks.** Each job uses approximately 7 GB of memory.


`coadd_maps.py`
---------------

This script is an example of the library. It does not use a command or
mapcat. It reads depth-1 FITS files from a directory, applies the
preprocessors to each map and merges the maps with `RhoKappaMapCoadder`.
Then it writes the rho and kappa maps of the coadd.

The paths in the script are for one data set. Change them before you use the
script.
