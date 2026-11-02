Analyze coadds with `sotrp`
===========================

The `sotrp` command runs the time-resolved pipeline on the coadds that
`sotrp-coadd` registered in mapcat. See [Coadds: overview](overview.md) for
the data flow.


Dependencies
------------

Before you run `sotrp` on coadds, make sure that you have these items:

1. **Registered coadds.** A `sotrp-coadd` run with
   `mapcat_registration.enabled: true` (see [Make coadds](make_coadds.md)).
2. **The coadd FITS files.** The `rho`, `kappa` and `time_mean` files of each
   coadd. `sotrp` does not use the depth-1 map files.
3. **The mapcat environment variables.** `MAPCAT_DATABASE_NAME` and
   `MAPCAT_DEPTH_ONE_COADD_PARENT` must have the same values as in the
   `sotrp-coadd` run. The coadd paths in mapcat are relative to
   `MAPCAT_DEPTH_ONE_COADD_PARENT`.
4. **The services that your config uses.** For example, SOCat for the source
   catalogs, and lightcurvedb or lightserve for the outputs.

### sqlmodel versions

sqlmodel 0.0.43 and later versions store datetime fields as UTC-aware values.
These versions do not accept naive datetimes. sotrplib is compatible with the
older and the newer versions:

- It gives UTC-aware datetimes to mapcat.
- It compares stored times with astropy `Time`. `Time` uses UTC for a naive
  value.

The two versions write SQLite files that you can use with each version.


Config
------

To read registered coadds, use the `mapcat_database` generator with
`map_type: "coadd_rhokappa"`. `sample_read_coadds.json` is a full example:

```json
"maps": {
    "map_generator_type": "mapcat_database",
    "map_type": "coadd_rhokappa",
    "instrument": "SOLAT",
    "frequency": "f090",
    "start_time": "2025-08-01T00:00:00",
    "end_time": "2026-01-01T00:59:59",
    "rerun": "True"
},
"preprocessors": [],
```

- The generator selects coadds by `frequency` and `start_time`/`end_time`.
  It compares these times with the start and stop times of each coadd.
- You can also select coadds by `coadd_type` and `map_ids`. For coadds, the
  `map_ids` are `coadd_id`s.
- The generator does not accept `array` or source-position filters. Mapcat
  has no tube_slot rows or sky-coverage rows for coadds.
- You can use the default `time_binning`, because each coadd is one full
  window.
- **Set `rerun` to true.** See "Processing status" below.
- **Do not add preprocessors.** `sotrp-coadd` applied them to each input map.
  You can use postprocessors (for example, `flatfield`).
- The default of `map_units` is Jy. The reader uses the `BUNIT` of the files
  if they have one.


Run the command
---------------

```
sotrp -c config.json
```

Use one job for each band. Each job analyzes the coadds of its band, one
after the other. A SLURM job script is the same as for depth-1 maps:

```
cd /path/to/sotrplib
source .venv/bin/activate
source env_setup   # MAPCAT_* (incl. MAPCAT_DEPTH_ONE_COADD_PARENT), socat, lightcurvedb
srun --overlap sotrp -c /path/to/f090_config.json > f090_sotrp.log 2>&1
```


What the pipeline does with a coadd
-----------------------------------

`CoaddRhoKappaMapReader` (`sotrplib/maps/database.py`) reads the
`depth_one_coadds` table. It gives one `CoaddRhoAndKappaMap` for each coadd:

- It finds the `rho`, `kappa` and `time_mean` files in
  `MAPCAT_DEPTH_ONE_COADD_PARENT`.
- It makes `array` from the `tube_slot`s of the linked depth-1 maps (for
  example, `i1i3i4i6`). The beam FWHM depends only on the frequency.
- It uses the time map without change, because the `time_mean` map of a
  coadd contains absolute unix times. A depth-1 time map contains seconds
  from the start of the observation.

Then the pipeline does the same steps as for a depth-1 map:

1. Build.
2. `finalize()` (flux and SNR from rho and kappa).
3. Postprocessors.
4. Pointing sources.
5. Forced photometry.
6. Source subtraction.
7. Blind search.
8. Sifter.
9. Outputs.

There are two differences from a depth-1 map:

- **Pointing:** the pipeline fits a pointing model on the coadd and uses it
  for that run. It does **not save** the model to mapcat, because the key of
  the pointing-residual table is the depth-1 `map_id`. The pipeline does not
  load depth-1 pointing models for a coadd. `sotrp-coadd` did not apply a
  pointing correction to the input maps.
- **Times:** the time of each measurement is the mean time, with hit weights,
  of the pixel in the coadd. This time can be anywhere in the window.


Processing status
-----------------

Mapcat records a processing status for each depth-1 map and each coadd in the
`time_domain_processing` table. There is one status row for each map or
coadd. The status of a coadd uses its `coadd_id`. The status of a depth-1 map
uses its `map_id`.

### The status values

| Status | Meaning | Set by |
|---|---|---|
| `processing` | A run started on the map or coadd. | The reader, when it reads the map. |
| `completed` | The run is complete. | The pipeline, at the end of the run. `sotrp-coadd`, when it registers a coadd. |
| `failed` | The run stopped with an error. | The pipeline, if an exception occurs. |
| `permafail` | The map is not usable. | A person, with `mapcatreset --status permafail`. |

### How `sotrp` uses the status

When the reader selects the maps or coadds, it does these checks:

1. It always skips a map or coadd with the status `permafail`.
2. If `rerun` is false, it skips a map or coadd with the status `completed`.
3. If `rerun` is false, it skips a map or coadd with the status `processing`
   that started less than 2 hours ago. After 2 hours, the reader reads the
   map again, because the old run probably stopped.
4. It sets the status of each map or coadd that it reads to `processing`.

At the end of the run, the pipeline sets the status to `completed`. If an
exception occurs, it sets the status to `failed`.

### Status of a coadd

`sotrp-coadd` sets the status of each coadd to `completed` when it registers
the coadd. Thus, a `sotrp` run on coadds must set `rerun: true`. If not,
`sotrp` skips all coadds.

### Status of the depth-1 maps in a coadd

By default, `sotrp-coadd` does not read or change the status of its input
depth-1 maps. There are two reasons:

- The status of a depth-1 map records the result of the `sotrp` run on that
  map.
- One map can be in many coadds (weekly, monthly). One status for each map
  cannot show this.

Thus, `sotrp-coadd` also uses the maps with the status `completed`, and it
does not overwrite the status that `sotrp` recorded. It always skips a map
with the status `permafail`. The links in `link_depth_one_map_to_coadd`
record the maps in each coadd.

To record a status for each input map, set `"track_processing": true` in the
`maps` config of `sotrp-coadd`. Then `sotrp-coadd` does these steps:

- It skips the maps with the status `completed`, if `rerun` is not set.
- It sets the status of each map that it reads to `processing`.
- It sets the status of each merged map to `completed`.
- It sets the status of each map that it did not merge to `failed`.
- If the run stops because of an error, it sets the status of each map that
  it read to `failed`.
