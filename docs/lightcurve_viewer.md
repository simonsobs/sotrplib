Viewing Pipeline Output with LightServe / LightView
====================================================

`sotrplib` can write forced-photometry output directly into
[`lightcurvedb`](https://github.com/simonsobs/lightcurvedb)'s parquet
backend (via `sotrplib.outputs.lightcurvedb.LightcurveDBOutput`) or over
HTTP to a running [`lightserve`](https://github.com/simonsobs/lightserve)
instance (via `sotrplib.outputs.lightserve.LightServeOutput`). Either way,
you end up with a directory of parquet files that
[`lightview`](https://github.com/simonsobs/lightview) can browse in a
webapp. This page documents the local setup used to run that stack
end-to-end: `sotrplib` -> parquet output -> `lightserve` (API) ->
`lightview` (frontend).

This is intended for local development / spot-checking pipeline output,
not a production deployment.

## 1. Pipeline output layout

Whichever output class you use, `lightcurvedb`'s parquet backend expects a
base directory shaped like this:

```
<output_dir>/
  sources.parquet          # one row per known source
  instruments.parquet      # optional; one row per instrument/frequency
  fluxes/
    <source_id>.parquet    # one file per source, all its flux measurements
  cutouts/
    <source_id>.parquet    # one file per source, its image cutouts
  unassigned_fluxes/
    ...                    # blind-search detections with no catalog match
```

If you're using `LightcurveDBOutput` directly, this is written under
`settings.parquet_base_path` on your `LightcurveDBSettings`. If you're
using `LightServeOutput`, this directory is owned and managed by whatever
`lightserve` instance you point the pipeline at.

## 2. Setting up `lightserve`

`lightserve` is a FastAPI app that wraps `lightcurvedb` and exposes it over
HTTP; `lightview` talks to it, and it's also what `LightServeOutput` writes
through when running the pipeline in "remote" mode.

```bash
git clone git@github.com:simonsobs/lightserve.git
cd lightserve
uv venv
uv pip install -e ".[dev,ephemeral]"
```

Run it against a directory of parquet output (either one `sotrplib` wrote
itself via `LightcurveDBOutput`, or one it's about to write to via
`LightServeOutput`):

```bash
source .venv/bin/activate

LIGHTCURVEDB_BACKEND_TYPE=parquet \
LIGHTCURVEDB_PARQUET_BASE_PATH=/path/to/output_dir \
TELEMETRY__ENABLE=false \
uvicorn lightserve.api:app --port 7777 --host 127.0.0.1
```

Notes:

- `LIGHTCURVEDB_PARQUET_BASE_PATH` must point at the `output_dir` described
  above (the parent of `sources.parquet`/`fluxes/`/`cutouts/`), not a
  specific file inside it.
- `TELEMETRY__ENABLE=false` disables telemetry reporting, which otherwise
  tries (and fails, slowly) to phone home when running somewhere without
  outbound network access, e.g. a compute node.
- Use `--host 127.0.0.1` if you're going to reach it via an SSH tunnel from
  the same machine (see below). Use `--host 0.0.0.0` if something else on
  the network needs to reach it directly -- this does not change what
  `127.0.0.1:<port>` resolves to locally, just whether other hosts can
  connect.
- If `lightserve`'s venv was built on a different host (e.g. a compute
  node with a different `$HOME` layout), it will not run elsewhere -- its
  `.venv/bin/python3` shebang bakes in an absolute path. Rebuild the venv
  (`rm -rf .venv && uv venv && uv pip install -e ".[dev,ephemeral]"`) on
  whichever host you're actually running it on.

Once it's up, `http://127.0.0.1:7777/docs` serves interactive API docs, and
is a quick way to sanity check it's alive and pointed at the right data
before starting `lightview`.

## 3. Running `sotrplib` with `LightServeOutput`

To have `sotrplib` write to a running `lightserve` instance instead of
writing parquet directly, add `LightServeOutput` to your pipeline's
`outputs`:

```python
from sotrplib.outputs.lightserve import LightServeOutput

outputs = [
    LightServeOutput(
        hostname="http://127.0.0.1:7777",
        # token_tag / identity_server only needed if lightserve is
        # running behind SOAuth; leave unset for a local, unauthenticated
        # instance.
    ),
]
```

`LightServeOutput` looks up each source's `socat_id` against
`lightserve`'s `/sources/` endpoint to translate it to `lightcurvedb`'s
internal `source_id`, then posts flux measurements and cutouts for it. A
source with no crossmatch, or no resolved `observation_mean_time`, is
skipped with a warning rather than raising.

## 4. Setting up `lightview`

`lightview` is the React/Vite frontend that visualizes what's in
`lightcurvedb` (via `lightserve`).

```bash
git clone git@github.com:simonsobs/lightview.git
cd lightview
```

It needs a recent Node (22.x). If you don't have `nvm` set up on this
host yet:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
# start a new shell, or `source ~/.nvm/nvm.sh`, then:
nvm install 22
nvm alias default 22
```

Then install dependencies and configure which `lightserve` to talk to:

```bash
nvm use 22
npm install
cp .env.development.sample .env.development
```

Edit `.env.development` so `VITE_SERVICE_URL` points at your running
`lightserve` instance, e.g.:

```
VITE_SERVICE_URL=http://127.0.0.1:7777
```

Then start the dev server:

```bash
npm run dev
```

By default Vite picks a port starting at 5173 and increments if it's
taken (in practice, on a shared login node with several dev servers
already running, you'll often land on something like 8002/8003 -- check
the terminal output for the actual port it bound).

## 5. Accessing it from your laptop

If `lightserve` and `lightview` are running on a remote host (e.g. a
cluster login node) rather than your own machine, forward both ports over
SSH:

```bash
ssh -L 8002:localhost:8002 -L 7777:localhost:7777 <user>@<host>
```

(swap `8002` for whatever port `lightview`'s dev server actually bound
to). Then open `http://localhost:8002` in your browser.

If the `ssh` command itself fails immediately with something like
`kex_exchange_identification: Connection closed by remote host`, that's a
network/connection-level failure, not an auth problem -- on Princeton
clusters this almost always means you're off-campus and not connected to
the VPN (GlobalProtect). Connect to VPN and retry.

## 6. Troubleshooting

- **500 error when opening a source in `lightview`**: check `lightserve`'s
  logs. A common cause is `extra` metadata that failed to round-trip
  through parquet correctly (fixed upstream in `lightcurvedb`, but only
  once you're on a version with that fix).
- **Cone search / "nearby sources" hangs or errors with real catalog
  positions**: usually an RA convention mismatch (`lightcurvedb` sources
  are stored in `[0, 360)`, and some older query paths assumed
  `(-180, 180]`). Also fixed upstream in `lightcurvedb`.
- **Everything looks dead but nothing crashed**: before assuming the
  backend/frontend processes died, rule out a frozen SSH terminal/tunnel
  first -- `curl` the `lightserve` port directly from the remote host; if
  that responds quickly, the servers are fine and it's your connection
  that's stuck.
- **Detections show a `1970-01-01` timestamp**: this was a real bug where
  a pixel with zero hits in the (unfiltered) time map was trusted
  directly, even though the (matched-filtered) flux/SNR maps can have a
  valid detection there. Fixed upstream in `sotrplib` by gap-filling
  zero-hit pixels from a local hits-weighted neighbor average before
  falling back to the observation start/mid/end time.
