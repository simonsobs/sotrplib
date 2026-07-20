"""
Dispatch historical multi-band/tube forced-photometry extraction for
repeat-confirmed transient candidates from aggregate_candidates.py
(Script A).

For all candidate rows together:
  1. Register each as a *persistent, monitored* source in the real
     production socat.db (not a throwaway temp db) -- so every future
     sotrp run picks it up automatically via SOCat.forced_photometry_sources(),
     with no further action needed per-candidate. This db is NOT what the
     sweep jobs below read from (see 3).
  2. Sync the newly-registered socat sources into lightcurvedb (via the
     existing `lightcurvedb-socat` console script), so lightcurvedb Source
     rows exist before the historical sweep starts producing flux
     measurements to attach to them.
  3. Dispatch slurm_wrapper_mapcat_historical_extractor.py once per batch
     (default: all candidates in one batch -- see --batch-size), pointed at
     a fresh db scoped to just that batch's candidates, not the production
     db from step 1. The wrapper generates one slurm job per (band, optics
     tube) that measures every source in its scoped db against every
     historical map in a single pass -- so cost scales with bands x tubes,
     not with candidate count x bands x tubes. Pointing the sweep at the
     production db instead would force every job to also force-photometer
     every other monitored source (tens of thousands, most unrelated) that
     happens to fall on each map. Pass --also-output-lightcurvedb to
     additionally backfill lightcurvedb with every historical epoch (needs
     a postgres reachable from the compute nodes); otherwise the sweep
     writes pickle output only.

Defaults to NOT submitting to SLURM (matching this repo's established
generate-then-explicitly-submit convention) -- pass --submit to also
chain a call to slurm_submitter.py once scripts are generated.
"""

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import astropy.units as u
import structlog
from astropy.coordinates import ICRS

log = structlog.get_logger()

_REQUIRED_COLUMNS = ["cluster_id", "ra_deg", "dec_deg", "flux_jy"]

_PASSTHROUGH_FLAGS = [
    "bands",
    "optics_tubes",
    "flux_threshold",
    "ncores",
    "nserial",
    "save_thumbnails",
    "thumbnail_radius",
    "group_name",
    "script_dir",
    "scratch_dir",
    "pythonpath",
    "also_output_lightcurvedb",
]


def read_candidate_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as fh:
        return list(csv.DictReader(fh))


def validate_rows(rows: list[dict]) -> list[dict]:
    """Hard-fail (not silently drop) on bad rows -- a human may have
    hand-edited the CSV between Script A and this, and a silently-dropped
    row is a silently-missed candidate."""
    seen_ids: set[str] = set()
    for i, row in enumerate(rows):
        for col in _REQUIRED_COLUMNS:
            if not row.get(col):
                raise ValueError(
                    f"Row {i} ({row.get('cluster_id')}): missing/empty '{col}'"
                )
        if row["cluster_id"] in seen_ids:
            raise ValueError(f"Duplicate cluster_id: {row['cluster_id']}")
        seen_ids.add(row["cluster_id"])
    return rows


def register_socat_sources(
    rows: list[dict], socat_db_path: Path, dry_run: bool
) -> None:
    if dry_run:
        for row in rows:
            print(
                f"[dry-run] would register socat source: {row['cluster_id']} "
                f"ra={row['ra_deg']} dec={row['dec_deg']} flux={row['flux_jy']} Jy "
                f"(monitored) in {socat_db_path}"
            )
        return

    from socat.client.db import Client

    client = Client(db_url=f"sqlite:///{socat_db_path}")
    ## re-runs of this dispatch (or the wrapper it chains to) must not
    ## re-create sources: socat has no name-uniqueness constraint, so a
    ## blind create_source duplicates every row. Query names straight
    ## from sqlite -- the client has no lookup-by-name.
    already_registered: set[str] = set()
    if socat_db_path.exists():
        import sqlite3

        try:
            with sqlite3.connect(socat_db_path) as connection:
                already_registered = {
                    row[0]
                    for row in connection.execute("SELECT name FROM fixed_sources")
                }
        except sqlite3.Error:
            already_registered = set()
    for row in rows:
        if row["cluster_id"] in already_registered:
            log.info(
                "dispatch_historical_extraction.register_socat_sources.already_present",
                cluster_id=row["cluster_id"],
            )
            continue
        source = client.create_source(
            position=ICRS(
                ra=float(row["ra_deg"]) * u.deg, dec=float(row["dec_deg"]) * u.deg
            ),
            name=row["cluster_id"],
            flux=float(row["flux_jy"]) * u.Jy,
            flags={"monitored": True},
        )
        log.info(
            "dispatch_historical_extraction.register_socat_sources.registered",
            cluster_id=row["cluster_id"],
            socat_source_id=str(source.source_id),
        )


def sync_lightcurvedb(socat_db_path: Path, dry_run: bool) -> None:
    cmd = ["lightcurvedb-socat"]
    if dry_run:
        print(
            f"[dry-run] would run: {' '.join(cmd)} (SOCAT_MODEL_DATABASE_NAME={socat_db_path})"
        )
        return

    env = os.environ.copy()
    env["socat_client_client_type"] = "db"
    env["socat_model_database_name"] = str(socat_db_path)
    log.info("dispatch_historical_extraction.sync_lightcurvedb.running")
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError:
        # Non-fatal: this is a best-effort optimization (the source row
        # existing before the sweep starts), not a hard requirement --
        # socat registration (the load-bearing "will keep being
        # monitored" behavior) must not be undone or blocked by a
        # lightcurvedb deployment being unreachable from wherever this
        # runs, and each sweep job's own --also-output-lightcurvedb
        # output will make its own attempt anyway (visible per-job in its
        # own logs if that also fails). Matches the SOCat-then-local-
        # ephemeris-fallback pattern already used in
        # sotrplib/maps/preprocessor.py's AsteroidMasker.
        log.warning(
            "dispatch_historical_extraction.sync_lightcurvedb.failed",
            message="Continuing without lightcurvedb sync -- socat "
            "registration and SLURM dispatch are not gated on this.",
            exc_info=True,
        )


def ensure_trailing_slash(path: str) -> str:
    """`Path` objects normalize away trailing slashes, but the wrapper
    script's own out_dir/slurm_script_dir join is plain string
    concatenation (not os.path.join) -- so out_dir must keep an explicit
    trailing slash as a *string*, all the way through, for that
    concatenation to land on a proper "<out_dir>/slurm_job_scripts/"
    rather than a "<out_dir>slurm_job_scripts/" run-together directory
    name. Wrapping it in Path() at any point before that concatenation
    would silently strip the slash back off."""
    return path if path.endswith("/") else path + "/"


def resolve_slurm_script_dir(out_dir: str) -> Path:
    """Replicates slurm_wrapper_mapcat_historical_extractor.py's own
    default out_dir/slurm_script_dir join (plain string concatenation,
    args.out_dir + args.slurm_script_dir -- not os.path.join), which only
    applies when --slurm-script-dir is left at its default
    ("slurm_job_scripts/"). This lets Script B know where the generated
    .slurm scripts land without parsing the wrapper's stdout (it prints
    paths, but has no machine-readable manifest). `out_dir` must already
    have a trailing slash (see ensure_trailing_slash) and must be a str,
    not a Path -- Path() would strip it right back off."""
    return Path(f"{out_dir}slurm_job_scripts/")


def chunk(seq: list, size: int) -> Iterator[list]:
    if size <= 0:
        yield seq
        return
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def build_wrapper_command(
    wrapper_script: Path,
    rows: list[dict],
    out_dir: str,
    sweep_socat_db_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    """`sweep_socat_db_path` is deliberately NOT `args.socat_db_path` (the
    real production db): the wrapper measures every source registered in
    whatever db it's pointed at against every historical map in one job per
    band x tube, so pointing it at production (tens of thousands of
    monitored sources) would force-photometer the whole field on every map,
    not just this batch's candidates -- one job per (band, tube) instead of
    one per (band, tube, source) only stays cheap if the sweep db is scoped
    to this batch."""
    cmd = [
        sys.executable,
        str(wrapper_script),
        "--ra",
        *[row["ra_deg"] for row in rows],
        "--dec",
        *[row["dec_deg"] for row in rows],
        "--flux",
        *[row["flux_jy"] for row in rows],
        "--source-id",
        *[row["cluster_id"] for row in rows],
        "--socat-db-path",
        str(sweep_socat_db_path),
        "--out-dir",
        out_dir,
    ]
    for flag in _PASSTHROUGH_FLAGS:
        value = getattr(args, flag)
        if value is None:
            continue
        cli_flag = "--" + flag.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(cli_flag)
        elif isinstance(value, list):
            cmd.append(cli_flag)
            cmd.extend(str(v) for v in value)
        else:
            cmd.extend([cli_flag, str(value)])
    return cmd


def invoke_wrapper(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would run: {' '.join(cmd)}")
        return
    log.info("dispatch_historical_extraction.invoke_wrapper.running", n_args=len(cmd))
    subprocess.run(cmd, check=True)


def submit_or_print_instructions(
    slurm_script_dir: Path, submit: bool, dry_run: bool, submitter_script: Path
) -> None:
    if dry_run:
        print(f"[dry-run] would submit scripts in {slurm_script_dir}")
        return
    if submit:
        log.info(
            "dispatch_historical_extraction.submit_or_print_instructions.submitting",
            slurm_script_dir=str(slurm_script_dir),
        )
        subprocess.run(
            [sys.executable, str(submitter_script), "--dir", str(slurm_script_dir)],
            check=True,
        )
    else:
        print(f"Generated slurm scripts in {slurm_script_dir}")
        print(f"To submit: python {submitter_script} --dir {slurm_script_dir}")


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        prog="dispatch-historical-extraction",
        description=(
            "Register aggregate_candidates.py's repeat-confirmed candidates "
            "as monitored socat sources, sync them into lightcurvedb, and "
            "dispatch the historical multi-band/tube forced-photometry sweep "
            "for each."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument(
        "--wrapper-script",
        type=Path,
        default=Path(__file__).parent / "slurm_wrapper_mapcat_historical_extractor.py",
    )
    parser.add_argument(
        "--submitter-script",
        type=Path,
        default=Path(__file__).parent.parent
        / "depth1_map_analysis"
        / "slurm_submitter.py",
    )
    parser.add_argument(
        "--socat-db-path",
        type=Path,
        default=Path("/scratch/gpfs/SIMONSOBS/users/amfoster/scratch/socat.db"),
        help="Real production socat.db -- NOT the wrapper's own default "
        "throwaway temp db. Candidates are registered here as persistent "
        "monitored sources.",
    )
    parser.add_argument("--min-snr", type=float, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Split candidates into chunks of this size, each dispatched as "
        "its own wrapper invocation. Default 0 processes all candidates in "
        "one chunk, which is what makes 'one job per band x tube' cheap -- "
        "with --batch-size set, each chunk generates its own independent "
        "band x tube job set into the same --out-dir, so later chunks' "
        "slurm scripts overwrite earlier chunks' filenames (only after the "
        "earlier chunk's jobs were already submitted, so --submit runs "
        "still work, but re-inspecting/resubmitting from --out-dir "
        "afterwards will only show the last chunk). Use a separate "
        "--out-dir per chunk if you need batching with resubmittable "
        "output.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)

    parser.add_argument("--bands", nargs="+", default=None)
    parser.add_argument("--optics-tubes", nargs="+", default=None)
    parser.add_argument("--flux-threshold", type=str, default=None)
    parser.add_argument("--ncores", type=int, default=None)
    parser.add_argument("--nserial", type=int, default=None)
    parser.add_argument(
        "--save-thumbnails", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--thumbnail-radius", type=str, default=None)
    parser.add_argument("--group-name", type=str, default=None)
    parser.add_argument("--script-dir", type=str, default=None)
    parser.add_argument("--scratch-dir", type=str, default=None)
    parser.add_argument(
        "--pythonpath",
        type=str,
        default=None,
        help="Exported as PYTHONPATH in each slurm job (e.g. a worktree "
        "checkout of sotrplib to run instead of the venv's editable install).",
    )
    parser.add_argument(
        "--also-output-lightcurvedb",
        action="store_true",
        default=False,
        help="Add the 'lightcurvedb' source_output to each sweep job. Off by "
        "default: it needs a lightcurvedb postgres reachable from the compute "
        "nodes, and an output failure fails the whole sotrp job (marking its "
        "maps 'failed' in mapcat). Pickle output always happens regardless.",
    )

    parser.add_argument("--submit", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    rows = validate_rows(read_candidate_csv(args.candidate_csv))
    if args.min_snr is not None:
        rows = [r for r in rows if float(r["max_snr"]) >= args.min_snr]
    if args.max_candidates is not None:
        rows = rows[: args.max_candidates]

    if not rows:
        print("No candidates survive filtering -- nothing to dispatch.")
        return

    out_dir = str(
        args.out_dir
        or (
            args.candidate_csv.parent
            / f"historical_extraction_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
    )
    out_dir = ensure_trailing_slash(out_dir)
    slurm_script_dir = resolve_slurm_script_dir(out_dir)

    log.info(
        "dispatch_historical_extraction.main.start",
        n_candidates=len(rows),
        out_dir=out_dir,
        dry_run=args.dry_run,
    )

    ## Registering in the real production db (for future ongoing monitoring)
    ## is deliberately separate from the db the sweep jobs themselves read
    ## from -- see build_wrapper_command's docstring. Each chunk gets its
    ## own scoped db file since the wrapper measures every source in
    ## whatever db it's given against every historical map; sharing one
    ## scoped db across chunks would have later chunks' jobs re-measure
    ## earlier chunks' sources too.
    register_socat_sources(rows, args.socat_db_path, dry_run=args.dry_run)
    sync_lightcurvedb(args.socat_db_path, dry_run=args.dry_run)

    n_chunks = 0
    for i, batch in enumerate(chunk(rows, args.batch_size)):
        sweep_socat_db_path = Path(out_dir) / f"scoped_historical_socat_{i}.db"
        cmd = build_wrapper_command(
            args.wrapper_script, batch, out_dir, sweep_socat_db_path, args
        )
        invoke_wrapper(cmd, dry_run=args.dry_run)
        submit_or_print_instructions(
            slurm_script_dir,
            submit=args.submit,
            dry_run=args.dry_run,
            submitter_script=args.submitter_script,
        )
        n_chunks += 1

    n_bands = len(args.bands) if args.bands else 4
    n_tubes = len(args.optics_tubes) if args.optics_tubes else 6
    log.info(
        "dispatch_historical_extraction.main.complete",
        n_candidates=len(rows),
        n_jobs=n_chunks * n_bands * n_tubes,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
