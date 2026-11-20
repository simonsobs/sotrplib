Make coadds with `sotrp-coadd`
==============================

The `sotrp-coadd` command (`sotrplib/coadd_cli.py`) makes one coadd from the
depth-1 maps that its config selects. See [Coadds: overview](overview.md) for
the data flow.

To make coadds for many time windows, use the helper script
`submit_week_coadds.py` (see [Helper scripts](scripts.md)). The script writes
the configs that this page describes.


Run the command
---------------

1. Set the environment variables (see "Environment" below).
2. Write a config file (see "Config" below).
3. Run the command:

   ```
   sotrp-coadd -c config.json
   ```

The command has one option, `-c`, for the config file.

Each input map takes approximately 5 to 6 minutes. The memory use is
approximately 7 GB, and it does not increase with the number of maps.


Environment
-----------

`sotrp-coadd` uses these mapcat environment variables:

```
export MAPCAT_DATABASE_NAME=/path/to/mapcat.sqlite
export MAPCAT_DEPTH_ONE_PARENT=/path/to/depth1/maps       # depth-1 paths are relative to this
export MAPCAT_DEPTH_ONE_COADD_PARENT=/path/to/coadds      # coadd paths are relative to this
```

Depth-1 maps and coadds have **different** root directories. Thus, you can
keep the coadds in your own data directory, and the depth-1 maps in a shared
directory.

- If `MAPCAT_DEPTH_ONE_COADD_PARENT` is not set, its value is the current
  directory. `sotrp-coadd` does not use `MAPCAT_DEPTH_ONE_PARENT` in its
  place.
- Each output directory must be in `MAPCAT_DEPTH_ONE_COADD_PARENT`.
  `sotrp-coadd` does this check before the coadd work starts. If the check
  fails, the error message gives the name of the variable.

The asteroid mask uses SOCat. Set these variables:

```
export socat_client_client_type=db
export socat_model_database_name=/path/to/socat.db
```


Config
------

`sotrp-coadd` reads a `CoaddSettings` JSON file (`sotrplib/config/coadd.py`).
`sample_coadd_config.json` is a full example for one week of f090 maps.

| Field | Meaning |
|---|---|
| `maps` | The depth-1 maps for the coadd. Use the `mapcat_database` generator with `map_type: "intensity"` (raw maps). See "Select the maps" below. |
| `preprocessors` | The preprocessors to apply, in sequence, to **each** depth-1 map before the merge. See "Preprocessors" below. |
| `map_coadder` | The method that merges the maps: `{"coadd_type": "rhokappa"}`. |
| `map_outputs` | The coadd fields to write to FITS, and the output directory. See "Outputs" below. |
| `mapcat_registration` | `coadd_name`, `coadd_type` and `enabled`. See "Registration in mapcat" below. |


Select the maps
---------------

The `maps` generator selects depth-1 maps from mapcat with these fields:

- `frequency` (necessary).
- `start_time` and `end_time`: the time window.
- `array` (optional): the maps of one array only. If you do not set it, the
  coadd contains the maps of all arrays of the band.
- `time_binning`: how the generator compares a map with the time window.

### Time binning

The observation of a depth-1 map can start in one window and stop in the
next window. The `time_binning` field sets the window that gets this map:

| Mode | The window includes a map if... | A map that crosses a window boundary... |
|---|---|---|
| `left-bound` | its `start_time` is in `[start, end)` | goes into one window only |
| `right-bound` | its `stop_time` is in `[start, end)` | goes into one window only |
| `restrictive` | all of the map is in the window | goes into no window |
| `loose` (default) | a part of the map is in the window | goes into the two windows |

For a set of adjacent windows, use `left-bound` or `right-bound`. Then each
map is in one coadd only, and there are no gaps.


Preprocessors
-------------

`sotrp-coadd` applies the preprocessors to each depth-1 map before the merge.
Use the same preprocessor configs as for `sotrp` (`sotrplib/config/preprocessors.py`).

This sequence is typical (see `sample_coadd_config.json`):

1. `planet_mask`: masks the planets (15 arcmin).
2. `asteroid_mask`: masks the asteroids from SOCat (10 arcmin). If SOCat is
   not available, it can use a local ephemeris file (`ephem_file_path`).
3. `matched_filter`: applies the matched filter and makes rho and kappa.
   Set `beam1d` to the 1D beam profile of the band.
4. `kappa_rho`: cleans kappa, and cuts the rho pixels that have a low kappa
   (`cut_on`, `fraction`).
5. `edge_mask`: masks the map edges on kappa (10 arcmin).

After the preprocessors, each map is a rho/kappa map.


How the maps are merged
-----------------------

`RhoKappaMapCoadder` adds `rho` and `kappa` over the union of the input
footprints. Thus, in the coadd:

- `flux = rho / kappa` is the mean flux, with inverse-variance weights.
- `snr = rho / sqrt(kappa)`.
- `hits` is the sum of the input hits.
- `time_mean` is the mean, with hit weights, of the **absolute** unix times
  of the inputs. The coadder calculates it for each pixel.
- `observation_start` is the earliest start of the inputs.
  `observation_end` is the latest end.
- `array` is the list of the different input arrays, sorted and without
  spaces (for example, `i1i3i4i6`).


Outputs
-------

Set the fields to write with `map_outputs[].fields`. You can request each of
these fields: `rho`, `kappa`, `flux`, `snr`, `hits` and `time_mean`.
`sotrp-coadd` writes `rho` and `kappa` before `finalize()`, and `flux` and
`snr` after `finalize()`.

The file names have the format `{frequency}_{array}_{start}_{field}.fits`,
for example `f090_i1i3i4i6_1758171214_flux.fits`. `{start}` is the unix time
of the **start of the earliest input map**. It is not the start of the window.

Each file with flux units records its unit in the FITS `BUNIT` header. The
matched filter uses mJy. Thus:

- `flux` has the unit `mJy`.
- `rho` has the unit `mJy-1`.
- `kappa` has the unit `mJy-2`.
- `snr`, `hits` and `time_mean` have no flux unit.

When sotrplib reads a map from disk, it gets the flux unit from `BUNIT`. If
the file has no `BUNIT`, sotrplib uses the `map_units` value from the config.


Registration in mapcat
----------------------

If `mapcat_registration.enabled` is true, `sotrp-coadd` writes these rows in
mapcat:

- `depth_one_coadds`: one row for the coadd. The row contains `coadd_name`,
  `coadd_type`, `frequency`, `start_time`, `stop_time`, `ctime` and the file
  paths. The paths are relative to `MAPCAT_DEPTH_ONE_COADD_PARENT`:
  `map_path` (flux), `rho_path`, `kappa_path` (also `ivar_path`) and
  `mean_time_path`.
- `link_depth_one_map_to_coadd`: one `(map_id, coadd_id)` row for each input
  map.
- `time_domain_processing`: a `completed` status row for the coadd.

The `depth_one_coadds` table has no array column. Thus, a registered coadd
has a frequency but no array.

If `enabled` is false, `sotrp-coadd` writes the FITS files only.

For the processing status of the coadd and of the input maps, see
[Analyze coadds: processing status](analyze_coadds.md#processing-status).


Errors in input maps
--------------------

If a map causes an error in the build, the preprocessors or the merge:

1. `sotrp-coadd` records the error and the traceback in the log
   (`stream_coadd.map_failed`).
2. It does not put the map in the coadd.
3. It continues with the next map.

At the end of the run, the `sotrp_coadd.maps_excluded` warning gives the
`map_id`s of these maps. If all maps fail, the run stops with an error, and
there is no coadd.

To find the maps to process again, compare the maps in the window with the
linked maps of the coadd (`link_depth_one_map_to_coadd`).
