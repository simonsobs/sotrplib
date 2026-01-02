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
We use `ruff` for formatting. When you go to commit your code, it will automatically be 
formatted thanks to the pre-commit hook.

Tests are performed using `pytest`.

## extra required packages

see pyproject.toml


## Setting up and Running the Pipeline

After following the development instructions above, you will be able to run the pipeline by running the following:

`sotrp -c [path to config file]`

The config file is a .json which contains a dictionary of all the handlers and inputs to the pipeline. 
You can see two examples: `sample.json` and `sample_read_map.json` in the top level directory.

A config file is read in by a basic handler (see `sotrplib/handlers/basic.py`), which de-serializes to Python objects using the
code in `sotrplib/config/config.py` and the relevant config files.

An example of the pipeline after de-serializing the real data loading example is given in `sotrplib/docs/act.md`. Here the only 
additional step is adding a distinct catalog for the forced photometry sources.

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

