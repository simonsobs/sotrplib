Mock real-time run
==================

A mock real-time run replays the depth-1 maps of a mapcat, one simulated day
at a time, through the full time-domain chain: mapcat, `sotrp`, socat,
lightcurvedb and lightview. Use it to test the chain before the real-time
system runs.

The scripts are in `scripts/mock_real_time/`:

| File | What it does |
|---|---|
| `mockrt.py` | The command. It makes the run, submits the jobs and records each step. |
| `feed.py` | Makes the live mapcat and copies the maps of one day into it. |
| `configs.py` | The run settings, the schedule and the generated configs and SLURM scripts. |


What a run does
---------------

The run has a live mapcat and a live socat. The live mapcat starts with no
maps. The live socat is a copy of a socat database.

For each simulated day (a "tick" job), `mockrt.py` does these steps:

1. **Feed.** It copies the maps that start in the day from the source
   mapcat to the live mapcat. Then it fills the sky coverage of the new maps
   (the mapcat `updatesky` function).
2. **Daily run.** It runs `sotrp` on the new maps of the day. All the bands
   and tubes are in one run, with the multi-array map matcher. The forced
   photometry goes to lightcurvedb. The socat update adds each confirmed
   map-match group to the live socat as a monitored source (see
   [Map matching](map_matching.md#add-confirmed-events-to-socat)). The
   forced photometry of the next days then measures these sources.
3. **Backfill.** For each new socat source, it runs the historical
   lightcurve extractor on all the live maps that start before the end of
   the day. The measurements go to lightcurvedb. lightcurvedb keeps
   duplicate measurements, so the ledger records each source that is done
   (`backfill:<socat_id>`), and a rerun does not measure it again.
4. **Week.** At the end of each week, it submits a weekly coadd job for each
   band, and a job that runs `sotrp` on the weekly coadds after them. The
   map matcher of the coadds uses two bands (`min_arrays: 1`,
   `min_bands: 2`). This job also does the socat update and the backfill.
5. **Month.** After the weekly coadds of the last week of a month, a job
   makes a monthly coadd of the weekly coadds of each band, and a job runs
   `sotrp` on the monthly coadds.
6. **Next day.** It submits the tick of the next day.

The weeks and months do not stop the ticks. They run at the same time, as in
the real-time system.

A ledger (`ledger.sqlite`) records the status of each step. A step that is
done does not run again. Thus, if a tick fails, correct the problem and
submit the tick again (`mockrt.py start --day N`).


Services
--------

The services job runs postgres (in apptainer), creates the lightcurvedb
tables and runs lightserve. It writes `services.env` with the host, the
port and the password. The jobs read this file (through `env.sh`). The
compute nodes connect to postgres by the host name of the services node.

Make the postgres image one time, on a login node (the compute nodes have
no network access):

```
apptainer pull postgres_16.sif docker://postgres:16
```

To see the lightcurves in lightview, run the lightview dev server on the
login node with `scripts/mock_real_time/lightview.config.mjs`. The vite
server sends `/api/*` to lightserve (a proxy), so the browser gets the page
and the data from one port:

```
cd $LIGHTVIEW_DIR
VITE_SERVICE_URL=/api \
LIGHTVIEW_DIR=$PWD \
LIGHTSERVE_TARGET=http://<services node>:<lightserve port> \
    npx vite --config <sotrplib>/scripts/mock_real_time/lightview.config.mjs
```

`LIGHTSERVE_TARGET` is `LIGHTSERVE_URL` in `services.env`. The lightserve
port changes each time the services job starts. Then, on your computer,
open a tunnel to the vite port only, and open http://localhost:6979/:

```
ssh -N -L 6979:localhost:6979 <user>@<login node>
```

Do not set `VITE_SERVICE_URL` to a second tunnel port (for example
`http://127.0.0.1:<lightserve port>`). The browser then sends the API
requests to that port on your computer. If that forward does not work (for
example, an old tunnel keeps the vite port), lightview shows 0 sources and
no error.

lightview uses vite 8, which needs Node.js 20.19 or later. With an earlier
Node.js, npm does not install the rolldown binding and vite stops with
"Cannot find native binding".


Procedure
---------

1. Make the run:

   ```
   python scripts/mock_real_time/mockrt.py init --run-dir /path/to/run \
     --source-mapcat /path/to/mapcat.sqlite \
     --source-socat /path/to/socat.db \
     --depth-one-parent /path/to/depth1/<run> \
     --code-dir /path/to/sotrplib/checkout --venv /path/to/.venv \
     --lightserve-venv /path/to/lightserve/.venv \
     --postgres-image /path/to/postgres_16.sif \
     --start-time 2025-09-04T00:00:00Z --n-days 37
   ```

   Examine `mockrt.json`. You can change the settings in this file, for
   example `tick_mem`, `pace_minutes` or `exclude_maps_csv`.
2. Start the services: `mockrt.py services --run-dir /path/to/run --submit`.
   Wait for `services.env`.
3. Start the run: `mockrt.py start --run-dir /path/to/run`.
4. Monitor the run: `mockrt.py status --run-dir /path/to/run`.

To test one day, feed the earlier days and run one tick with no next tick
and no coadds:

```
python scripts/mock_real_time/mockrt.py feed --run-dir /path/to/run --first 0 --last 5
python scripts/mock_real_time/mockrt.py tick --run-dir /path/to/run --day 6 --no-next --no-coadds
```

Run the tick in a job: it needs the resources of a daily run.


The run directory
-----------------

```
<run-dir>/
  mockrt.json  env.sh  services.env  ledger.sqlite
  mapcat.sqlite  socat.db                  the live databases
  coadds/weekly/<YYYYMMDD>/  coadds/monthly/<YYYYMMDD>/
  days/day<NNN>_<YYYYMMDD>/   config.json sotrp.log pickles/ map_match/ socat_updates/ backfill/
  weeks/week<NN>_<YYYYMMDD>/  coadd_configs/ config.json sotrp.log ...
  months/month<NN>_<YYYYMMDD>/
  postgres/  slurm/
```


Schedule
--------

- Day `d` is `[start + d days, start + d + 1 days)`. A map is in the day of
  its start time.
- Week `k` is the days `7k` to `7k + 6`. The last day of the run ends its
  week, also if the week is short.
- A month is `month_weeks` (default 4) weeks. The run makes a monthly coadd
  only for a full month.


Limits
------

- The live mapcat and the live socat are SQLite files. The ticks and the
  coadd jobs write to them at the same time. SQLite waits for locks, but a
  very long write can cause a "database is locked" error.
- socat keeps only the position, the flux, the name and the flags of a
  source. The map-match data of each new source is in
  `socat_updates/socat_update_<start>.json`.
- The coadd measurements are in the same lightcurvedb sources as the
  depth-1 measurements. Their module is the list of tubes of the coadd
  (for example `i1i3i4i6`).
