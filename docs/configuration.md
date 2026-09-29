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


Outputs
-------

There is no default output. Add the outputs that you need:

- `source_outputs`: the measured sources. The `output_type` is `pickle`,
  `json`, `cutout`, `lightcurvedb` or `lightserve`.
- `map_outputs`: the map fields as FITS files (`output_type: "maps"`).

For example:

```json
"source_outputs": [
    {"output_type": "pickle", "directory": "."}
]
```

The code for the outputs is in `sotrplib/outputs/`.
