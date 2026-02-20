# sotrplib
Simons Observatory Time Resolved Pipeline Library

A pipeline to ingest fits maps, perform pre- and post-processing, forced photometry and blind searching for point sources.

Currently, the output is a pandas database in pickle format.

See `scripts/end-to-end/` for an example of a full pipeline run including socat and lightcurvedb

## Development requirements

To get ready for development, create a virtual enviroment and install the package:
```
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```
If you don't have uv installed, you can install it with `pip install uv`, or
just go ahead and use `pip install -e ".[dev]"`. 

We use `ruff` for formatting. When you go to commit your code, it will automatically be 
formatted thanks to the pre-commit hook.

Tests are performed using `pytest`.

## extra required packages
Any required packages should be listed in `pyproject.toml`

If a package is missing, you can manually install it with `uv install [package]`. Please then report this on the GitHub
[issue tracker](https://github.com/simonsobs/sotrplib/issues).


## Setting up and Running the Pipeline

After following the development instructions above, you will be able to run the pipeline by running the following:

`sotrp -c [path to config file]`

The config file is a .json which contains a dictionary of all the pipeline segments and inputs. 
You can see several examples in the top level directory: `sample_*.json` 

A config file is read in by a basic handler (see `sotrplib/handlers/basic.py`), which de-serializes to Python objects using the
code in `sotrplib/config/config.py` and the relevant config files.

### Source Catalog (socat)

We have implemented the source catalog as a mock version of `socat` (https://github.com/simonsobs/socat/).
To make things work with an old ACT type catalog, socat includes a runnable script `socat-act-fits` which ingests the .fits file into a pickle file which can be converted into a mock socat object in the pipeline.

Thus to run the pipeline with an ACT source catalog, you first create the socat pickle file:

`socat-act-fits -f [act_catalog.fits] -o socat.pickle`

Then tell socat where to find it:

`export socat_client_client_type=pickle`
`export socat_client_pickle_path=socat.pickle`

Setting the sotrplib config to use `socat` as one of it's source catalogs:

```
"source_catalogs": [
        {
            "catalog_type": "socat",
            "flux_lower_limit": "0.01 Jy"
        }
    ],
```
then generates a mock socat client given the info above.

This source catalog can then be used for forced photometry, pointing, blind-search crossmatching, etc.

### Map Catalog (mapcat)

One way to ingest maps into the pipeline is to manually add them into the config, like the example `sample_read_unfiltered_map.json` . 
This is convenient for testing a specific map, or a one-off, etc. 

However, running on a full set of maps and keeping track of map metadata, etc. requires a map database.
We call this `mapcat` (https://github.com/simonsobs/mapcat) and again have an ingestion script for ACT-like map sets.

To ingest ACT depth1 maps into a mapcat sqlite db, you would run the script `actingest` after setting the relevant mapcat environment variables; 

```
export MAPCAT_DEPTH_ONE_PARENT=/path/to/depth1/maps
export MAPCAT_DATABASE_NAME=/path/to/mapcat.sqlite
```
This tells the map catalog where to look for the maps and where the database lives. 

With the existance of a mapcat database, the pipeline can be configured to read from there via :

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
for example, which tells the runner to read in 1 map at f090, from array i6 and to rerun it if it has already been analyzed.

### How to configure the pipeline

The .json config file is from where the pipeline runner reads.
Allowed methods and their properties can be accessed in the `sotrplib/config/` directory.
Each file contains the relevant configurations and required methods for each type of object; i.e. maps, preprocessors, forced_photometries, etc.
These configurations are read-in, converted from JSON to pydantic models (`sotrplib/config/config.py`), and used by the pipeline handler to construct the pipeline.
The basic handler can be found in `sotrplib/handlers/base.py`

Let's follow one example through from .json config to understand what is happening.
We'll use maps.
The pipeline expects maps to be a list of `ProcessableMap` objects.
You'll notice in the samples there are two different settings for `maps`; a dictionary or a list of dictionaries.
If the converting function sees a list, it knows that they are lists of map objects, so it processes each one individually.
If the conversion sees a dictionary it knows to expect a map_generator, which, in the case of mapcat_database, it will query the `mapcat.sqlite` db and construct a list of map objects.

Let's take the case of `sample_read_unfiltered_map.json`. Here we have 

```json
"maps": [
  {
    "map_type": "inverse_variance",
    "intensity_map_path": "./depth1_1538613353_pa5_f090_map.fits",
    "weights_map_path": "./15386/depth1_1538613353_pa5_f090_ivar.fits",
    "time_map_path": "./depth1_1538613353_pa5_f090_time.fits",
    "frequency": "f090",
    "band": "pa5",
    "intensity_units": "K",
    "box": [
      {
        "ra": {
          "value": 138.52,
          "unit": "deg"
        },
        "dec": {
          "value": -13.095,
          "unit": "deg"
        }
      },
      {
        "ra": {
          "value": 140.52,
          "unit": "deg"
        },
        "dec": {
          "value": -11.095,
          "unit": "deg"
        }
      }
    ]
  }
],
```
so we can see `map_type` is `inverse_variance`. Going to `config/maps.py`, you can find where map_type is inverse_variance; i.e. the `InverseVarianceMapConfig` class.
You can see what the required / default arguments are and what the pipeline does when it converts that input `to_map` -- it creates a ProcessableMap class of subclass IntensityAndInverseVarianceMap.

If you look at the other example, `sample_read_mapcat.json`, you will see 

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
which clearly shows `map_generator_type` as the descriptor, not `map_type`. This implies that it will generate maps from the source (which is listed as mapcat_databse here).
Checking `config/maps.py` we see the subclass with that map_generator_type is `MapCatDatabaseConfig` which returns a `MapCatDatabaseReader` instance; returning a list of map objects corresponding to what is configured.

Once the map objects are loaded, the pipeline handler then builds the maps and injects them into the rest of the pipeline.

The various other components of the pipeline are built in a similar manner, for example map preprocessing is built by configuring a list of `PreProcessor` objects, etc.

The pipeline then runs as per your config, and the steps in the handler script.

### Pipeline Outputs

The current default is to output to pickle files because these are simply converted from the pydantic models transferred between the pipeline components in production mode.

The output format can be found in `outputs/core.py`, and in the default case is the `PickleSerializer`.

Essentially this is just dictionaries of lists of MeasuredSource objects (and InjectedSource objects in the case you're simulating sources).

These MeasuredSource objects contain information about their measurement and even cutouts.


### Running with prefect

[prefect](https://docs.prefect.io/v3/get-started) is a workflow orchestrator that provides a conveneient web interface for monitoring and running the pipeline.
Installing and invoking the pipeline using prefect follows the same basic pattern as above.

```console
uv sync --extra prefect
source .venv/bin/activate
export sotrp_runner=prefect
sotrp -c [path to config file]
```

This will start a temporary prefect server, if you want a persistent server you can start one as described in the [prefect docs](https://docs.prefect.io/v3/get-started/quickstart#open-source).

The runner can also be specified via the configuration file or as a command-line argument.

```console
prefect server start --host [HOSTNAME, e.g., localhost] --port [PORT, e.g., 8899] --background
```

This will start a prefect server and provide a URL to the dashboard, in this case http://localhost:8484.
`sotrp-prefect` can then be invoked either by manually specifying the `PREFECT_API_URL` as an environment variable, e.g.,

```console
PREFECT_API_URL=http://localhost:8484/api sotrp-prefect -c [path to config file]
```

or by using the prefect tool

```console
prefect config set PREFECT_API_URL=http://localhost:8484/api
```

The server can be stopped with

```console
prefect server stop
```

