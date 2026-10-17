"""
The run settings, the schedule and the generated configs of a mock
real-time run.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_BEAM = (
    "/scratch/gpfs/SIMONSOBS/users/amfoster/so/sotrplib/"
    "profile_{band}_1756699200_20000000000.txt"
)


@dataclass
class RunConfig:
    """The settings of a run. Saved as <run_dir>/mockrt.json."""

    run_dir: str
    source_mapcat: str
    "The mapcat with all the depth-1 maps to replay. Read only."
    source_socat: str
    "The socat database to start from. The run uses a copy."
    depth_one_parent: str
    "MAPCAT_DEPTH_ONE_PARENT of the depth-1 maps."
    code_dir: str
    "The sotrplib checkout (or worktree) that the jobs run."
    start_time: str
    "Start of day 0 (ISO, UTC). The weeks and the months start at this time."
    n_days: int
    "The run has the observations in [start_time, start_time + n_days)."
    venv: str = ""
    "Python environment of the jobs. Default: <code_dir>/.venv."
    lightserve_venv: str = ""
    "Python environment with lightserve and lightcurvedb (services job)."
    bands: list[str] = field(default_factory=lambda: ["f090", "f150", "f220", "f280"])
    tubes: list[str] = field(
        default_factory=lambda: ["i1", "i3", "i4", "i6", "c1", "i5"]
    )
    exclude_maps_csv: str | None = None
    "Maps to leave out (see feed.read_exclude_list)."
    week_days: int = 7
    month_weeks: int = 4
    "A month is this number of weeks. The monthly coadd merges their coadds."
    beam_profile: str = DEFAULT_BEAM
    "Beam profile of each band; {band} is replaced."
    runner: str = "prefect"
    blind_snr: float = 3.0
    high_sig: float = 5.0
    low_sig: float = 3.0
    lightcurvedb: bool = True
    "Write the forced photometry to lightcurvedb (needs the services job)."
    pace_minutes: float = 0.0
    "Minimum wall time between the starts of two ticks. 0: no wait."
    sequential: bool = True
    """If true, the run does one step at a time: after the last observation
    of a week, the next tick starts when the week (and month) analysis is
    done. If false, the coadd jobs run at the same time as the ticks."""
    account: str = "simonsobs"
    tick_cpus: int = 8
    tick_mem: str = "128G"
    tick_time: str = "04:00:00"
    coadd_cpus: int = 4
    coadd_mem: str = "64G"
    coadd_time: str = "08:00:00"
    services_time: str = "7-00:00:00"
    postgres_image: str = ""
    "Apptainer image of postgres (apptainer pull postgres_16.sif docker://postgres:16)."
    postgres_port: int = 0
    "0: choose a port from the SLURM job ID."
    lightserve_port: int = 0
    "0: postgres port + 1."
    planet_ephemeris: str = (
        "/scratch/gpfs/SIMONSOBS/users/amfoster/so/sotrplib/de440s.bsp"
    )
    "The planet mask reads de440s.bsp from the working directory (the code "
    "directory). The jobs link this file there if it is not there."

    # --- paths in the run directory ---------------------------------------
    @property
    def root(self) -> Path:
        return Path(self.run_dir)

    @property
    def live_mapcat(self) -> Path:
        return self.root / "mapcat.sqlite"

    @property
    def live_socat(self) -> Path:
        return self.root / "socat.db"

    @property
    def coadd_parent(self) -> Path:
        return self.root / "coadds"

    @property
    def slurm_dir(self) -> Path:
        return self.root / "slurm"

    @property
    def observations_file(self) -> Path:
        return self.root / "observations.json"

    @property
    def ledger(self) -> Path:
        return self.root / "ledger.sqlite"

    @property
    def env_file(self) -> Path:
        return self.root / "env.sh"

    @property
    def services_env(self) -> Path:
        return self.root / "services.env"

    @property
    def python(self) -> str:
        return str(Path(self.venv or Path(self.code_dir) / ".venv") / "bin" / "python")

    # --- schedule ----------------------------------------------------------
    @property
    def t0(self) -> datetime:
        t = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)

    @property
    def end(self) -> datetime:
        return self.t0 + timedelta(days=self.n_days)

    def observations(self) -> list[datetime]:
        """The start times of the observations of the run, in time order
        (observations.json, which mockrt.py writes)."""
        return [
            datetime.fromisoformat(t)
            for t in json.loads(self.observations_file.read_text())
        ]

    def obs_window(self, obs: int) -> tuple[datetime, datetime]:
        """A window that has the start time of the maps of the observation,
        and no other maps (the observations are hours apart)."""
        start = self.observations()[obs].replace(microsecond=0)
        return start, start + timedelta(minutes=1)

    def day_of(self, t: datetime) -> int:
        return (t - self.t0) // timedelta(days=1)

    def week_of(self, day: int) -> int:
        return day // self.week_days

    def week_ending_at(self, obs: int) -> int | None:
        """The week that ends with this observation (the next observation
        is in a later week, or this is the last observation), or None."""
        times = self.observations()
        week = self.week_of(self.day_of(times[obs]))
        if obs == len(times) - 1 or self.week_of(self.day_of(times[obs + 1])) != week:
            return week
        return None

    def week_window(self, week: int) -> tuple[datetime, datetime]:
        start = self.t0 + timedelta(days=week * self.week_days)
        end = min(
            start + timedelta(days=self.week_days),
            self.t0 + timedelta(days=self.n_days),
        )
        return start, end

    def month_of(self, week: int) -> int:
        return week // self.month_weeks

    def ends_month(self, week: int) -> int | None:
        """The month that ends with this week, or None. Only full months:
        the run does not make a monthly coadd of a short month."""
        if (week + 1) % self.month_weeks == 0:
            end_day = (week + 1) * self.week_days
            if end_day <= self.n_days:
                return self.month_of(week)
        return None

    def month_window(self, month: int) -> tuple[datetime, datetime]:
        days = self.week_days * self.month_weeks
        start = self.t0 + timedelta(days=month * days)
        return start, start + timedelta(days=days)

    def obs_dir(self, obs: int) -> Path:
        start, _ = self.obs_window(obs)
        return self.root / "obs" / f"obs{obs:04d}_{start:%Y%m%dT%H%M%S}"

    def week_dir(self, week: int) -> Path:
        start, _ = self.week_window(week)
        return self.root / "weeks" / f"week{week:02d}_{start:%Y%m%d}"

    def month_dir(self, month: int) -> Path:
        start, _ = self.month_window(month)
        return self.root / "months" / f"month{month:02d}_{start:%Y%m%d}"

    # --- io ------------------------------------------------------------------
    def save(self) -> Path:
        path = self.root / "mockrt.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, run_dir: str | Path) -> "RunConfig":
        return cls(**json.loads((Path(run_dir) / "mockrt.json").read_text()))


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- sotrp configs ------------------------------------------------------------


def _beams(run: RunConfig) -> dict[str, str]:
    return {b: run.beam_profile.format(band=b) for b in run.bands}


def _common(run: RunConfig) -> dict:
    """The parts of the sotrp config that the depth-1 and coadd runs share."""
    return {
        "pointing_provider": {
            "photometry_type": "lmfit_pointing",
            "thumbnail_half_width": "0.1 deg",
            "min_flux": "0.3 Jy",
            "reproject_thumbnails": True,
            "allowable_centroid_offset": "3.0 arcmin",
        },
        "pointing_residual": {
            "pointing_residual_type": "median",
            "min_snr": 5.0,
            "min_sources": 10,
        },
        "postprocessors": [
            {
                "postprocessor_type": "flatfield",
                "sigma_val": 5.0,
                "tile_size": "0.5 deg",
            }
        ],
        "source_subtractor": {"subtractor_type": "photutils"},
        "blind_search": {
            "search_type": "photutils",
            "parameters": {"sigma_threshold": run.blind_snr},
        },
        "forced_photometry": {
            "photometry_type": "lmfit",
            "reproject_thumbnails": True,
            "flux_limit_centroid": "0.1 Jy",
            "thumbnail_half_width": "4 arcmin",
            "allowable_center_offset": "1.0 arcmin",
            "near_source_rel_flux_limit": 1.0,
        },
        "sifter": {
            "sifter_type": "default",
            "min_match_radius": "5.0 arcmin",
            "cuts": {"snr": [run.blind_snr, "inf"]},
        },
        "runner": run.runner,
    }


def _outputs(run: RunConfig, out_dir: Path) -> dict:
    source_outputs = [{"output_type": "pickle", "directory": str(out_dir / "pickles")}]
    if run.lightcurvedb:
        source_outputs.append({"output_type": "lightcurvedb", "upsert_sources": True})
    return {
        "source_outputs": source_outputs,
        "map_match_outputs": [
            {
                "output_type": "json",
                "directory": str(out_dir / "map_match"),
                "include_all_groups": True,
            }
        ],
    }


def _socat_with_update(out_dir: Path) -> list[dict]:
    """socat, with the update that adds confirmed map-match groups."""
    return [
        {
            "catalog_type": "socat",
            "update": {
                "add_confirmed_transients": True,
                "duplicate_radius": "1 arcmin",
                "name_prefix": "SO-T",
                "record_directory": str(out_dir / "socat_updates"),
            },
        }
    ]


def obs_config(run: RunConfig, obs: int) -> dict:
    """sotrp config of the depth-1 maps of one observation (all the tubes
    and the bands). The map matcher matches only the maps of the
    observation."""
    start, end = run.obs_window(obs)
    out_dir = run.obs_dir(obs)
    return {
        "maps": {
            "map_generator_type": "mapcat_database",
            "map_type": "intensity",
            "start_time": iso(start),
            "end_time": iso(end),
            "time_binning": "left-bound",
            "track_processing": True,
        },
        "source_catalogs": _socat_with_update(out_dir),
        "preprocessors": [
            {"preprocessor_type": "planet_mask", "mask_radius": "15 arcmin"},
            {
                "preprocessor_type": "matched_filter",
                "beam1d": _beams(run),
                "band_height": "1 deg",
                "shrink_holes": "5 arcmin",
                "noisemask_lim": 0.1,
                "noisemask_radius": "10 arcmin",
                "apod_holes": "10 arcmin",
            },
            {"preprocessor_type": "kappa_rho"},
            {
                "preprocessor_type": "edge_mask",
                "mask_on": "kappa",
                "edge_width": "10.0 arcmin",
            },
        ],
        "map_matcher": {
            "matcher_type": "multi_array",
            "radius": "1.5 arcmin",
            "min_arrays": 2,
            "high_sig": run.high_sig,
            "low_sig": run.low_sig,
        },
        **_common(run),
        **_outputs(run, out_dir),
    }


def coadd_analysis_config(
    run: RunConfig, coadd_type: str, start: datetime, end: datetime, out_dir: Path
) -> dict:
    """
    sotrp config of the coadds of one window (all bands in one run). The
    coadds are already filtered: no preprocessors. A coadd merges the tubes
    of a band, so the map matcher confirms an event with two bands.
    """
    return {
        "maps": {
            "map_generator_type": "mapcat_database",
            "map_type": "coadd_rhokappa",
            "coadd_type": coadd_type,
            "start_time": iso(start),
            "end_time": iso(end),
            "time_binning": "left-bound",
            "track_processing": True,
            # Registration sets the coadd status to completed.
            "rerun": True,
        },
        "source_catalogs": _socat_with_update(out_dir),
        "preprocessors": [],
        "map_matcher": {
            "matcher_type": "multi_array",
            "radius": "1.5 arcmin",
            "min_arrays": 1,
            "min_bands": 2,
            "high_sig": run.high_sig,
            "low_sig": run.low_sig,
        },
        **_common(run),
        **_outputs(run, out_dir),
    }


# --- environment and SLURM --------------------------------------------------------


def env_text(run: RunConfig) -> str:
    return f"""# Environment of the jobs of the mock real-time run in {run.root}.
export MAPCAT_DATABASE_NAME={run.live_mapcat}
export MAPCAT_DEPTH_ONE_PARENT={run.depth_one_parent}
export MAPCAT_DEPTH_ONE_COADD_PARENT={run.coadd_parent}
export socat_client_client_type=db
export socat_model_database_name={run.live_socat}
export TELEMETRY__ENABLE=false
# lightcurvedb (postgres) of the services job, if it is up.
if [ -f {run.services_env} ]; then source {run.services_env}; fi
"""


PREFECT_ENV = """# prefect server local to this job (node-local tmp, per-job port)
export TMPDIR=${TMPDIR:-/tmp}/$USER/prefect_${SLURM_JOB_ID:-local}
mkdir -p $TMPDIR
export PREFECT_HOME=$TMPDIR/prefect
mkdir -p $PREFECT_HOME
PORT=$((6000 + ${SLURM_JOB_ID:-969} % 2000))
export PREFECT_API_URL=http://127.0.0.1:$PORT/api
export PREFECT_SERVER_API_HOST=127.0.0.1
export PREFECT_SERVER_API_PORT=$PORT
export PREFECT_TASKS_RUNNER_PROCESS_POOL_MAX_WORKERS=${SLURM_CPUS_PER_TASK:-4}
prefect server start --host 127.0.0.1 --port $PORT < /dev/null > $PREFECT_HOME/server.log 2>&1 &
PREFECT_SERVER_PID=$!
until curl -s $PREFECT_API_URL/health > /dev/null; do sleep 1; done
trap 'kill $PREFECT_SERVER_PID 2>/dev/null' EXIT
"""


def slurm_script(
    run: RunConfig,
    name: str,
    command: str,
    cpus: int,
    mem: str,
    time: str,
    prefect: bool = False,
) -> str:
    """A SLURM script that runs `command` in the code directory with the run
    environment."""
    venv = Path(run.python).parent.parent
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH -A {run.account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={run.slurm_dir}/%x_%j.out

cd {run.code_dir}
[ -e de440s.bsp ] || ln -s {run.planet_ephemeris} de440s.bsp
source {venv}/bin/activate
source {run.env_file}
{PREFECT_ENV if prefect else ""}
{command}
"""


def services_script(run: RunConfig) -> str:
    """
    The SLURM job that runs postgres (apptainer), creates the lightcurvedb
    tables and runs lightserve. It writes services.env with the connection
    settings, which env.sh reads. Compute nodes connect to it by host name.
    """
    lightserve_venv = run.lightserve_venv or str(Path(run.code_dir) / ".venv")
    pg_port = run.postgres_port or "$((20000 + SLURM_JOB_ID % 20000))"
    ls_port = run.lightserve_port or "$((PGPORT + 1))"
    return f"""#!/bin/bash
#SBATCH --job-name=mockrt_services
#SBATCH -A {run.account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time={run.services_time}
#SBATCH --output={run.slurm_dir}/%x_%j.out

set -e
ROOT={run.root}
IMAGE={run.postgres_image}
PGDATA=$ROOT/postgres/data
PGRUN=$ROOT/postgres/run
mkdir -p $PGDATA $PGRUN
PGPORT={pg_port}
LSPORT={ls_port}
HOST=$(hostname)
# Keep the password of the first start (the data directory keeps it).
if [ -f $ROOT/postgres/password ]; then
    PGPASS=$(cat $ROOT/postgres/password)
else
    PGPASS=$(head -c 18 /dev/urandom | base64 | tr -dc 'A-Za-z0-9')
    (umask 077; echo $PGPASS > $ROOT/postgres/password)
fi

apptainer run --cleanenv \\
    -B $PGDATA:/var/lib/postgresql/data -B $PGRUN:/var/run/postgresql \\
    --env POSTGRES_PASSWORD=$PGPASS,POSTGRES_DB=lightcurvedb,PGPORT=$PGPORT \\
    $IMAGE > $ROOT/postgres/postgres.log 2>&1 &
PG_PID=$!
trap 'kill $LS_PID $PG_PID 2>/dev/null; rm -f {run.services_env}' EXIT

until apptainer exec $IMAGE pg_isready -q -U postgres -h 127.0.0.1 -p $PGPORT; do
    kill -0 $PG_PID || {{ echo "postgres stopped"; exit 1; }}
    sleep 2
done
echo "postgres is up on $HOST:$PGPORT"

export LIGHTCURVEDB_BACKEND_TYPE=postgres
export LIGHTCURVEDB_POSTGRES_HOST=$HOST
export LIGHTCURVEDB_POSTGRES_PORT=$PGPORT
export LIGHTCURVEDB_POSTGRES_USER=postgres
export LIGHTCURVEDB_POSTGRES_PASSWORD=$PGPASS
export LIGHTCURVEDB_POSTGRES_DB=lightcurvedb
export TELEMETRY__ENABLE=false
export FEED_FREQUENCY=90
export FEED_MODULE=i1

source {lightserve_venv}/bin/activate
lightcurvedb-setup
uvicorn lightserve.api:app --host 0.0.0.0 --port $LSPORT > $ROOT/postgres/lightserve.log 2>&1 &
LS_PID=$!

(umask 077; cat > {run.services_env} <<EOF
export LIGHTCURVEDB_BACKEND_TYPE=postgres
export LIGHTCURVEDB_POSTGRES_HOST=$HOST
export LIGHTCURVEDB_POSTGRES_PORT=$PGPORT
export LIGHTCURVEDB_POSTGRES_USER=postgres
export LIGHTCURVEDB_POSTGRES_PASSWORD=$PGPASS
export LIGHTCURVEDB_POSTGRES_DB=lightcurvedb
export LIGHTSERVE_URL=http://$HOST:$LSPORT
EOF
)
echo "lightserve is up on http://$HOST:$LSPORT"
wait $PG_PID
"""
