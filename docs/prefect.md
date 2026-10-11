Run `sotrp` with the prefect runner
===================================

The `sotrp` command has two runners. The `runner` field of the config selects
the runner:

| `runner` | Class | Code | How it runs the maps |
|---|---|---|---|
| `basic` (default) | `PipelineRunner` | `sotrplib/handlers/basic.py` | One map after the other, in one process. |
| `prefect` | `PrefectRunner` | `sotrplib/handlers/prefect.py` | Each map in a separate worker process, in parallel. |

The two runners do the same steps. The steps are in `BaseRunner`
(`sotrplib/handlers/base.py`). The prefect runner also shows each step on the
[prefect](https://docs.prefect.io/v3/get-started) dashboard.

Use the prefect runner to analyze many maps in one job, for example all the
depth-1 maps of one day. The runner needs all the maps of a run at the same
time for [map matching](map_matching.md).


How the prefect runner runs a pipeline
--------------------------------------

`BaseRunner.run()` is a prefect flow. The flow does these steps:

1. The map coadder groups the maps (`group_maps()`). With the default
   coadder, each map is one group.
2. The flow sends each group to a worker process (`ProcessPoolTaskRunner`).
   The worker builds the map, applies the preprocessors and does the
   photometry, the blind search and the sifter. The worker writes the map
   outputs (FITS files).
3. The worker returns a `MapResult` (`sotrplib/sifter/map_matching.py`) to
   the main process. A `MapResult` contains the candidates and the map data
   that the next steps need. It does not contain the map, so it is small.
4. When all the workers are complete, the map matcher groups the transient
   candidates across the maps. See [Map matching](map_matching.md).
5. The main process writes the source outputs of each map. Then it sets the
   processing status of each map to `completed`.

The main process keeps the `MapResult` of each map in memory until step 5.
Thus, the source outputs of a map contain the results of the map matching.


Requirements
------------

- Install the `prefect` extra:

  ```console
  uv sync --extra prefect
  ```

- Each object that the config makes must be serializable with `cloudpickle`.
  The runner sends the objects to the worker processes. For example,
  `SOCat` removes its database client before serialization and connects
  again in the worker. The worker gets the `socat_*` environment variables
  from the main process.
- The workers use the working directory of the command. Put the files that
  the library reads from the working directory in that directory. For
  example, the planet mask reads `de440s.bsp` from the working directory.
  If the file is not there, skyfield tries to download it.


Run on your computer
--------------------

See "Run with prefect" in the [README](../README.md#run-with-prefect). If you
do not start a prefect server, `sotrp` starts a temporary server.


Run on a SLURM compute node
---------------------------

Start a prefect server in each job. A compute node has no network access,
and a server of a different job can use the same port.

### 1. Make an environment file

Write a file, for example `env_prefect`, that you source in the job:

```bash
#!/bin/bash
# Database and catalog settings for the run.
export MAPCAT_DEPTH_ONE_PARENT='/path/to/depth1/'
export MAPCAT_DATABASE_NAME='/path/to/mapcat.sqlite'
export socat_client_client_type=db
export socat_model_database_name='/path/to/socat.db'
export TELEMETRY__ENABLE=false

export sotrp_runner=prefect

# Keep the prefect files on the node, one directory for each job.
export TMPDIR=${TMPDIR:-/tmp}/$USER/prefect_${SLURM_JOB_ID:-local}
mkdir -p $TMPDIR
export PREFECT_HOME=$TMPDIR/prefect
mkdir -p $PREFECT_HOME

# Use a different port for each job, so that jobs on one node do not share a server.
PORT=$((6000 + ${SLURM_JOB_ID:-969} % 2000))
export PREFECT_API_URL=http://127.0.0.1:$PORT/api
export PREFECT_SERVER_API_HOST=127.0.0.1
export PREFECT_SERVER_API_PORT=$PORT

# The process pool uses all the cores of the node by default, not the cores of the job.
export PREFECT_TASKS_RUNNER_PROCESS_POOL_MAX_WORKERS=${SLURM_CPUS_PER_TASK:-4}

# Start the server without a terminal. Then wait until it is ready.
prefect server start --host 127.0.0.1 --port $PORT < /dev/null > $PREFECT_HOME/server.log 2>&1 &
PREFECT_SERVER_PID=$!
until curl -s $PREFECT_API_URL/health > /dev/null; do sleep 1; done
```

**Set `PREFECT_TASKS_RUNNER_PROCESS_POOL_MAX_WORKERS`.** If you do not set
it, the process pool starts one worker for each core of the node. Each worker
keeps one map in memory, so the job can use more memory than it has.

### 2. Make a SLURM script

```bash
#!/bin/bash
#SBATCH --job-name=sotrp_prefect_day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# The working directory must contain de440s.bsp (see "Requirements").
cd /path/to/sotrplib
source .venv/bin/activate
source /path/to/env_prefect

sotrp -c /path/to/config.json
STATUS=$?

kill $PREFECT_SERVER_PID
exit $STATUS
```

### 3. Select the number of workers and the memory

Each worker keeps one depth-1 map and its filtered maps in memory. For the
lat-iso depth-1 maps (0.5 GB to 0.8 GB of FITS files for each map), these
jobs completed:

| Maps | Bands | Workers | Time | Peak memory |
|---|---|---|---|---|
| 8 | 1 | 4 | 5 min | 16 GB |
| 4 | 1 | 4 | 3 min | 14 GB |
| 24 | 4 | 8 | 9 min | 29 GB |

Use approximately 4 GB for each worker, plus the memory of the main process.


Analyze all the maps of one day in one job
------------------------------------------

Map matching compares only the maps of one run. To match detections across
the bands, put all the bands in one run:

1. In `maps`, do not set `frequency` or `array`. Set `start_time` and
   `end_time` to the day.
2. In the `matched_filter` preprocessor, give one beam profile for each band:

   ```json
   {
       "preprocessor_type": "matched_filter",
       "beam1d": {
           "f090": "/path/to/profile_f090.txt",
           "f150": "/path/to/profile_f150.txt",
           "f220": "/path/to/profile_f220.txt",
           "f280": "/path/to/profile_f280.txt"
       }
   }
   ```

   The preprocessor selects the profile with the `frequency` of each map. If
   a band has no profile, the preprocessor raises `ValueError`.
3. Add a `map_matcher` (see [Map matching](map_matching.md)).
4. Set `"runner": "prefect"`.


Processing status
-----------------

With a mapcat reader, the runner writes the status of each map in the
`time_domain_processing` table of mapcat:

- The reader sets `processing` when it reads the map.
- The main process sets `completed` after it writes the source outputs of
  the map.
- If the analysis or the source outputs of a map raise an exception, the
  runner sets `failed`.

The reader does not read a map with the status `completed`. To analyze the
maps again, set `"rerun": true` in `maps`.


Problems and solutions
----------------------

| Error | Cause | Solution |
|---|---|---|
| `TypeError: cannot pickle 'weakref.ReferenceType' object` | An object of the runner is not serializable, for example a database client. | Remove the object before serialization, as `SOCat` does (`__getstate__` and `__setstate__`). |
| `cannot download https://ssd.jpl.nasa.gov/.../de440s.bsp` | The file is not in the working directory, and the node has no network access. | Put `de440s.bsp` in the working directory of the job. |
| The job uses too much memory, or SLURM stops it. | There are too many workers. | Set `PREFECT_TASKS_RUNNER_PROCESS_POOL_MAX_WORKERS`. |
| The server does not start, or the flow uses the server of a different job. | Two jobs on one node use the same port. | Calculate the port from `SLURM_JOB_ID`. |
