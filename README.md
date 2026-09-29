# sotrplib
Simons Observatory Time Resolved Pipeline Library

sotrplib is a Python library for time-domain analysis of SO maps. Its classes
and functions read FITS maps and apply pre- and post-processing. They also do
forced photometry and a blind search for point sources, and write the results.

The package also installs two commands that use the library to run pipelines
from a JSON config:

- `sotrp`: runs the time-resolved pipeline on maps.
- `sotrp-coadd`: makes coadds of depth-1 maps and registers them in mapcat.

The `scripts/` directory has scripts that are not installed. Some scripts
write configs and SLURM jobs for the commands. Other scripts use the library
directly. See `scripts/end_to_end/` for a full pipeline run with socat and
lightcurvedb.

See [docs/overview.md](docs/overview.md) for the library modules, the
commands and the scripts, and [docs/](docs/README.md) for all documentation.

## Install

We like the package `uv` for managing packages and installing repos.
If you don't have it you can `pip install uv` or just pull it from their site 
`curl -LsSf https://astral.sh/uv/install.sh | sh`

sotrplib requires Python 3.12 or later. Make a virtual environment and install
the package:

```
uv venv --python=3.12
source .venv/bin/activate
uv pip install sotrplib
```

if you plan to develop, you should install the dev requirements:

`uv pip install -e ".[dev]"`

The pre-commit hook formats your code with `ruff` when you commit. The tests
use `pytest`.

## Missing packages

`pyproject.toml` lists all the necessary packages. If a package is missing,
install it with `uv pip install [package]`. Then report the problem on the
GitHub [issue tracker](https://github.com/simonsobs/sotrplib/issues).

## Run the pipeline

After you install the package, run the pipeline with the `sotrp` command:

```
sotrp -c [path to config file]
```
but the default config expects environment variables for  the
source catalog (`socat`) and the map catalog (`mapcat`).

- To make coadds, use the `sotrp-coadd` command (see
  [docs/coadding/](docs/coadding/overview.md)).
- To use the library in your own Python code, see [docs/act.md](docs/act.md).

The config file is a JSON file with the settings for each part of the
pipeline. The top directory has example configs (`sample_*.json`).
`sotrplib/cli.py` reads the file into the `Settings` model
(`sotrplib/config/config.py`). The model makes the library objects and gives
them to a runner (`sotrplib/handlers/`). See
[docs/configuration.md](docs/configuration.md) for the config fields and the
examples.

### Source catalog (socat)

sotrplib uses [socat](https://github.com/simonsobs/socat/) for the source
catalog. socat installs commands that add catalogs to its database. For
example, `socat-act-fits` adds an ACT FITS catalog. socat can also add
solar-system object ephemerides from a JPL Horizons parquet file. See the
socat README.

Set the socat environment variables:

```
export socat_client_client_type=db
export socat_model_database_name=socat.db
```

Then use one socat catalog in the config:

```
"source_catalogs": [
    {
        "catalog_type": "socat"
    }
],
```

The `socat` catalog type gets the type and the name of the database from the
environment variables.

You can also load a catalog file directly into a `RegisteredSourceCatalog`
(`sotrplib/source_catalog/core.py`). To do this, make a custom
`SourceCatalog` and add a config model for it. See
`sotrplib/source_catalog/source_catalog.py` for examples. We recommend the
socat database.

### Map catalog (mapcat)

You can give maps directly in the config, as in
`sample_read_unfiltered_map.json`. This is useful to test one map.

For a full set of maps, use the [mapcat](https://github.com/simonsobs/mapcat)
database. mapcat keeps the metadata of each map. To add ACT depth-1 maps to a
mapcat SQLite database, set these environment variables and run the `actingest`
command:

```
export MAPCAT_DEPTH_ONE_PARENT=/path/to/depth1/maps
export MAPCAT_DATABASE_NAME=/path/to/mapcat.sqlite
```

The first variable gives the root directory of the depth-1 maps. The second
variable gives the database file.

`sotrp-coadd` stores the coadd paths relative to a different root directory.
Thus, the coadds can be in a different directory from the depth-1 maps (for
example, your own data directory):

```
export MAPCAT_DEPTH_ONE_COADD_PARENT=/path/to/coadds
```

To read maps from mapcat, use the `mapcat_database` map generator in the
config:

```json
"maps": {
  "map_generator_type": "mapcat_database",
  "number_to_read": 1,
  "instrument": "SOLAT",
  "frequency": "f090",
  "array": "i6",
  "rerun": "True"
},
```

This example reads one f090 map of array i6. With `rerun`, it also reads a
map that the pipeline processed before.

### Pipeline outputs

There is no default output. Set the outputs in the config:

- `source_outputs`: the measured sources. The output types are `pickle`,
  `json`, `cutout`, `lightcurvedb` and `lightserve`.
- `map_outputs`: the map fields as FITS files.

The code for the outputs is in `sotrplib/outputs/`. The `pickle` output
(`PickleSerializer`) writes dictionaries of lists of `MeasuredSource`
objects. If you simulate sources, it also writes the `InjectedSource`
objects. A `MeasuredSource` contains the measurement and a cutout.

A `MeasuredSource` or a `RegisteredSource` can have a list of `CrossMatch`
objects (`sotrplib/sources/sources.py`). Each `CrossMatch` is a match to a
catalog. To find a source by its identifier (for example, in socat or
lightcurvedb), use `CrossMatch.catalog_idx`. Do not use
`CrossMatch.source_id`:

- `catalog_idx` is the unique, stable identifier of the source in the
  catalog (for example, a socat UUID).
- `source_id` can be a name (for example, "Ceres" for a solar-system
  object). It is not always unique.

### Run with prefect

[prefect](https://docs.prefect.io/v3/get-started) is a workflow orchestrator.
It has a web interface to monitor and run the pipeline.

1. Install the `prefect` extra:

   ```console
   uv sync --extra prefect
   source .venv/bin/activate
   ```

2. Run `sotrp` with the prefect runner:

   ```console
   export sotrp_runner=prefect
   sotrp -c [path to config file]
   ```

   You can also set `"runner": "prefect"` in the config file.

This procedure starts a temporary prefect server. To use a persistent server,
do these steps:

1. Start the server (see the
   [prefect docs](https://docs.prefect.io/v3/get-started/quickstart#open-source)):

   ```console
   prefect server start --host localhost --port 8484 --background
   ```

   The dashboard is at http://localhost:8484.

2. Set `PREFECT_API_URL` to the server. Use an environment variable:

   ```console
   PREFECT_API_URL=http://localhost:8484/api sotrp_runner=prefect sotrp -c [path to config file]
   ```

   Or use the prefect command:

   ```console
   prefect config set PREFECT_API_URL=http://localhost:8484/api
   ```

3. When you are done, stop the server:

   ```console
   prefect server stop
   ```

To run the prefect runner in a SLURM job, and to analyze all the bands of a
day in one job, see [Run `sotrp` with the prefect runner](docs/prefect.md).
To group the transient candidates across arrays and bands, see
[Map matching](docs/map_matching.md).
