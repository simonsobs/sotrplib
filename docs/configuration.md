# Configuration

Each command reads a JSON config file. A Pydantic Settings model in the
library reads the file and makes the library objects:

| Command | Model | Code |
|---|---|---|
| `sotrp` | `Settings` | `sotrplib/config/config.py` |
| `sotrp-coadd` | `CoaddSettings` | `sotrplib/config/coadd.py` |

The models for each part of the pipeline (maps, preprocessors, forced
photometry and others) are in `sotrplib/config/`. Each model has a method
(for example, `to_generator()` or `to_preprocessor()`) that makes the library
object. If you use the library directly, you can make these objects without
a config.

The configuration can change. More information will be added when it is
stable.


Examples
--------

The top directory of the repository has these example configs:

| File | Command | What it shows |
|---|---|---|
| `sample_read_unfiltered_map.json` | `sotrp` | One depth-1 map from FITS files, with a sky box. |
| `sample_read_mapcat.json` | `sotrp` | Depth-1 maps from mapcat. |
| `sample_read_act_mapcat.json` | `sotrp` | ACT depth-1 maps from mapcat. |
| `sample_read_coadds.json` | `sotrp` | Registered coadds from mapcat. |
| `sample_sim_map.json` | `sotrp` | A simulated map. |
| `sample_coadd_config.json` | `sotrp-coadd` | One weekly coadd. |

**Examine the field names carefully.** The models ignore a field that they do
not know (`extra="ignore"`). Thus, a field with an incorrect name has no
effect, and there is no error.


How `sotrp` uses the config
---------------------------

`sotrp -c config.json` does these steps (`sotrplib/cli.py`):

1. `Settings.from_file()` reads the JSON file into the `Settings` model.
2. `Settings.to_runner()` makes the library objects with
   `Settings.to_dependencies()`. For example, each item in `preprocessors`
   becomes a `MapPreprocessor`.
3. `Settings.to_runner()` gives these objects to the runner. The `runner`
   field selects `PipelineRunner` (`basic`, the default, in
   `sotrplib/handlers/basic.py`) or `PrefectRunner` (`prefect`, in
   `sotrplib/handlers/prefect.py`).
4. The command makes the maps from the `maps` field (see "Maps" below).
5. The runner runs the pipeline on each map. The steps for one map are in
   `BaseRunner` (`sotrplib/handlers/base.py`).
6. The map matcher groups the transient candidates of all the maps (see
   [Map matching](map_matching.md)).
7. The runner writes the source outputs of each map.

You can also set each top-level field with an environment variable that
starts with `sotrp_`, for example `sotrp_runner=prefect`.


Maps
----

The `maps` field is a list of maps or one map generator.

### A list of maps

Each item is one map. The `map_type` field selects the model. For example,
`sample_read_unfiltered_map.json` has this item:

```json
"maps": [
    {
        "map_type": "inverse_variance",
        "intensity_map_path": "./depth1_1538613353_pa5_f090_map.fits",
        "weights_map_path": "./depth1_1538613353_pa5_f090_ivar.fits",
        "time_map_path": "./depth1_1538613353_pa5_f090_time.fits",
        "frequency": "f090",
        "array": "pa5",
        "intensity_units": "K",
        "sky_box": [
            {"ra": {"value": 138.52, "unit": "deg"}, "dec": {"value": -13.095, "unit": "deg"}},
            {"ra": {"value": 140.52, "unit": "deg"}, "dec": {"value": -11.095, "unit": "deg"}}
        ]
    }
]
```

`map_type: "inverse_variance"` selects `InverseVarianceMapConfig`
(`sotrplib/config/maps.py`). Its `to_map()` method makes an
`IntensityAndInverseVarianceMap`. `sky_box` gives the lower-left and the
upper-right corners of the region to read.

### A map generator

The `map_generator_type` field selects the generator. For example,
`sample_read_mapcat.json` has this generator:

```json
"maps": {
    "map_generator_type": "mapcat_database",
    "number_to_read": 1,
    "instrument": "SOLAT",
    "frequency": "f090",
    "array": "i6",
    "rerun": "True"
}
```

`map_generator_type: "mapcat_database"` selects `MapCatDatabaseConfig`
(`sotrplib/config/maps.py`). Its `to_generator()` method makes a
`MapCatDatabaseReader`. This example reads one f090 map of array i6 from
mapcat. With `rerun`, the reader also reads a map that the pipeline
processed before.

The `map_type` field of the generator selects the type of map in mapcat:
`intensity` (default), `rhokappa`, `flux` or `coadd_rhokappa` (see
[Analyze coadds](coadding/analyze_coadds.md)).


Runner
------

The `runner` field selects the runner:

- `basic` (default): `PipelineRunner` analyzes one map after the other.
- `prefect`: `PrefectRunner` analyzes the maps in parallel worker processes.

See [Run `sotrp` with the prefect runner](prefect.md).


Beam profiles for several bands
-------------------------------

The `beam1d` field of the `matched_filter` preprocessor is one file, or one
file for each band:

```json
"beam1d": {
    "f090": "/path/to/profile_f090.txt",
    "f150": "/path/to/profile_f150.txt"
}
```

With one file for each band, one config can filter the maps of all the
bands. The preprocessor selects the file with the `frequency` of the map. If
a band has no file, the preprocessor raises `ValueError`.


Noisy regions of the map
------------------------

The matched filter uses one 2D noise spectrum for the full map. Some maps
have regions with much more noise than the remaining map, for example scan
stripes from the end of an observation. These regions make the noise spectrum
too large. Thus, the snr rms of these regions is more than 1, and the snr rms
of the remaining map is less than 1.

To remove the noisy regions from the noise spectrum, set
`noise_mask_max_rms` in the `matched_filter` preprocessor:

```json
{
    "preprocessor_type": "matched_filter",
    "noise_mask_max_rms": 1.2
}
```

The preprocessor then filters the map two times:

1. The first filter gives the snr map. The preprocessor calculates the robust
   rms of the snr in tiles of `noise_mask_tile_size` (default 0.5 deg).
2. The tiles with an rms more than `noise_mask_max_rms` are noisy. The
   preprocessor also marks `noise_mask_grow_tiles` (default 1) tiles around
   each noisy tile. It apodizes the edge of this region by `noise_mask_apod`
   (default 10 arcmin).
3. The second filter calculates the noise spectrum without the noisy region,
   and filters the full map with this spectrum.

If no tile is noisy, the second filter does not occur. If the noisy region is
more than `noise_mask_max_fraction` (default 0.5) of the map, the second
filter also does not occur, because the noise spectrum needs sufficient area.

After the second filter, the snr rms of the noisy region is still more than
1. To remove these tiles from the analysis, use the `snr_rms_mask`
postprocessor. Put it before the `flatfield` postprocessor, because the
flatfield makes the rms approximately 1 in all tiles:

```json
"postprocessors": [
    {"postprocessor_type": "snr_rms_mask", "max_rms": 1.2},
    {"postprocessor_type": "flatfield"}
]
```

The `snr_rms_mask` postprocessor sets the mask, the snr and the flux of the
map to 0 in the tiles where the snr rms is more than `max_rms` or less than
`min_rms` (default: no lower limit). Forced photometry and the blind search
do not use these tiles.


Sifter cuts
-----------

The `default` sifter moves a candidate to `noise_candidates` if a value of
the candidate is outside a cut. The `cuts` field changes one or more cuts:

```json
"sifter": {
    "sifter_type": "default",
    "cuts": {"snr": [3.0, "inf"]}
}
```

Each cut is `[min, max]`. The cuts that you do not give keep their default
values: `fwhm` `[0.5, 2.5]` (in units of the beam FWHM), `snr`
`[5.0, inf]` and `observation_mean_time` `[1, inf]`. See
`DEFAULT_SIFTER_CUTS` in `sotrplib/sifter/core.py`.


Map matching
------------

The `map_matcher` field groups the transient candidates across the maps of
a run. The default (`"matcher_type": "empty"`) does no map matching. See
[Map matching](map_matching.md). To confirm a 5-sigma detection with
3-sigma detections in other arrays, see "Use a low blind-search threshold"
in that document.


Outputs
-------

There is no default output. Add the outputs that you need:

- `source_outputs`: the measured sources. The `output_type` is `pickle`,
  `json`, `cutout`, `lightcurvedb` or `lightserve`.
- `map_outputs`: the map fields as FITS files (`output_type: "maps"`).
- `map_match_outputs`: the groups of the map matcher, one time for each run
  (`output_type: "json"`). See [Map matching](map_matching.md).

For example:

```json
"source_outputs": [
    {"output_type": "pickle", "directory": "."}
]
```

The code for the outputs is in `sotrplib/outputs/`.

The runner writes the source outputs after map matching. Thus, the source
outputs contain the results of the map matching.
