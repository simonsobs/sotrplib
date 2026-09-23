"""
Convert a `pickle_to_parquet`-flattened table into lightcurvedb's ingest
schema, and (optionally) actually push it through a lightcurvedb `Backend`
to prove the shape is correct.

`forced_photometry` and `transient_candidates` rows are eligible -- both
resolve to a stable identity to key a lightcurvedb `Source` on:
  - `forced_photometry` rows always carry their own `source_id` (a socat
    catalog name, e.g. "ACT-S J0058.5+0620").
  - `transient_candidates` rows have `source_id = None` (they're blind
    detections); instead they resolve to either the sifter's own
    crossmatch (a known source caught flaring -- reuses the *same*
    catalog name, and thus the same registered Source, as its
    forced_photometry measurements) or, if uncrossmatched, the pipeline's
    own auto-generated transient name (e.g. "SO-T J023444+0757.9").
`source_candidates`/`noise_candidates` rows have neither and are skipped.

Flux measurements (and, where a thumbnail survived flattening, cutouts) are
built as client-side objects with a `measurement_id` generated up front via
`uuid7`, and pushed through `Backend.fluxes.create_batch`/
`Backend.cutouts.create_batch` rather than `Backend.fluxes.ingest_dataframe`
-- that bulk path generates and discards its own `measurement_id`s
server-side, which would make it impossible to link a `Cutout` to the
measurement it belongs to. This mirrors how
`sotrplib.outputs.lightcurvedb.LightcurveDBOutput` does it for live runs.
A row only gets a `Cutout` if `pickle_to_parquet.py` was run with
`--keep-thumbnails` *and* that particular source had thumbnail data.

`source_id` must be a lightcurvedb `Source.source_id`, not sotrplib's own
`source_id` string (a catalog name like "ACT-S J0058.5+0620") -- so sources
have to be registered (or looked up) first and their name mapped to the
resulting UUID.
"""

import argparse as ap
import asyncio
import json
import sqlite3
from pathlib import Path

import pandas as pd
import uuid7
from lightcurvedb.config import Settings as LightcurveDBSettings
from lightcurvedb.models.cutout import Cutout
from lightcurvedb.models.flux import FluxMeasurement
from lightcurvedb.models.source import Source
from structlog import get_logger

log = get_logger()

ELIGIBLE_CATEGORIES = ("forced_photometry", "transient_candidates")


def resolve_source_identity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `_identity_name`/`_identity_socat_id` columns: own `source_id` if
    set (forced_photometry, and any already-named transient), else the
    sifter's crossmatch identity (a transient caught matching a known
    source), so both categories can be grouped into `Source`s uniformly --
    and a source flaring under both paths collapses onto the same Source.
    """
    names = []
    socat_ids = []
    for source_id, n_crossmatches, crossmatches in zip(
        df["source_id"], df["n_crossmatches"], df["crossmatches"]
    ):
        if pd.notna(source_id):
            names.append(source_id)
            socat_ids.append(None)
        elif n_crossmatches:
            cm = json.loads(crossmatches)[0]
            names.append(cm["source_id"])
            socat_ids.append(cm.get("catalog_idx"))
        else:
            names.append(None)
            socat_ids.append(None)

    out = df.copy()
    out["_identity_name"] = names
    out["_identity_socat_id"] = socat_ids

    unresolved = out["_identity_name"].isna().sum()
    if unresolved:
        log.warning(
            "lightcurvedb_export.dropping_unresolvable_rows",
            n=int(unresolved),
            reason="no source_id and no crossmatch",
        )
        out = out[out["_identity_name"].notna()]

    return out


def load_socat_name_to_id(socat_db_path: Path) -> dict[str, str]:
    """Map registered-source name -> socat UUID string, straight from
    socat's sqlite `fixed_sources` table (SSOs aren't in there, so not
    every forced_photometry source_id will resolve -- that's expected)."""
    if not Path(socat_db_path).exists():
        return {}
    with sqlite3.connect(socat_db_path) as connection:
        rows = connection.execute(
            "SELECT name, source_id FROM fixed_sources"
        ).fetchall()
    return {name: source_id for name, source_id in rows}


def build_lightcurvedb_sources(
    df: pd.DataFrame, socat_name_to_id: dict[str, str]
) -> tuple[list[Source], dict[str, str]]:
    """
    One `Source` per unique resolved identity in `df` (see
    `resolve_source_identity`), position taken as that identity's mean
    *measured* ra/dec (the detection's own position, not the crossmatch's).
    Returns the sources plus a name -> lightcurvedb-source_id (str UUID)
    map for `build_flux_measurement_frame`.
    """
    sources = []
    name_to_lc_id = {}
    for name, group in df.groupby("_identity_name"):
        socat_id = group["_identity_socat_id"].dropna()
        socat_id = socat_id.iloc[0] if len(socat_id) else socat_name_to_id.get(name)
        source = Source(
            socat_id=socat_id,
            name=name,
            ra=float(group["ra"].mean()),
            dec=float(group["dec"].mean()),
            variable=False,
        )
        sources.append(source)
        name_to_lc_id[name] = str(source.source_id)

    n_linked = sum(1 for s in sources if s.socat_id is not None)
    log.info(
        "lightcurvedb_export.built_sources",
        n_sources=len(sources),
        n_linked_to_socat=n_linked,
        n_unlinked=len(sources) - n_linked,
    )
    return sources, name_to_lc_id


def build_flux_measurements(
    df: pd.DataFrame, name_to_lc_id: dict[str, str]
) -> list[FluxMeasurement]:
    """One `FluxMeasurement` per eligible row (see `resolve_source_identity`),
    with a `measurement_id` generated here (rather than left to the backend)
    so `build_cutouts` can link a matching `Cutout` to it by ID."""
    missing = set(df["_identity_name"]) - set(name_to_lc_id)
    if missing:
        raise ValueError(f"{len(missing)} source_id(s) have no lightcurvedb mapping")

    err_ra_col = df.get("err_ra", pd.Series([None] * len(df), index=df.index))
    err_dec_col = df.get("err_dec", pd.Series([None] * len(df), index=df.index))

    measurements = []
    for (
        identity_name,
        frequency,
        module,
        time_unix,
        ra,
        dec,
        err_ra,
        err_dec,
        flux,
        err_flux,
    ) in zip(
        df["_identity_name"],
        df["frequency"],
        df["array"],
        df["observation_mean_time_unix"],
        df["ra"],
        df["dec"],
        err_ra_col,
        err_dec_col,
        df["flux"],
        df["err_flux"],
    ):
        measurements.append(
            FluxMeasurement(
                measurement_id=uuid7.create(),
                frequency=round(frequency),
                module=module,
                source_id=name_to_lc_id[identity_name],
                time=pd.Timestamp(time_unix, unit="s", tz="utc").to_pydatetime(),
                ra=float(ra),
                dec=float(dec),
                ra_uncertainty=float(err_ra) if pd.notna(err_ra) else None,
                dec_uncertainty=float(err_dec) if pd.notna(err_dec) else None,
                flux=float(flux) / 1000.0,  # mJy -> Jy
                flux_err=float(err_flux) / 1000.0,  # mJy -> Jy
                extra=None,
            )
        )
    return measurements


def _parse_thumbnail(raw) -> list[list[float]] | None:
    """`pickle_to_parquet.py --keep-thumbnails` JSON-encodes each source's
    thumbnail array into this column; rows converted without that flag (or
    sources that never had a thumbnail) leave it as None/NaN."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    return json.loads(raw)


def build_cutouts(
    df: pd.DataFrame, flux_measurements: list[FluxMeasurement]
) -> list[Cutout]:
    """One `Cutout` per eligible row that has thumbnail data (requires
    `pickle_to_parquet.py --keep-thumbnails`), linked to its matching
    `FluxMeasurement` by `measurement_id`. `df` and `flux_measurements` must
    be row-aligned (i.e. `flux_measurements` was built from this exact `df`
    via `build_flux_measurements`)."""
    if "thumbnail" not in df.columns:
        log.warning(
            "lightcurvedb_export.no_thumbnail_column",
            reason="discovery parquet was built without --keep-thumbnails",
        )
        return []

    unit_col = df.get("thumbnail_unit", pd.Series([None] * len(df), index=df.index))

    cutouts = []
    for raw_thumbnail, unit, fm in zip(df["thumbnail"], unit_col, flux_measurements):
        data = _parse_thumbnail(raw_thumbnail)
        if data is None:
            continue
        cutouts.append(
            Cutout(
                measurement_id=fm.measurement_id,
                data=data,
                time=fm.time,
                units=unit if isinstance(unit, str) else "mJy",
                frequency=fm.frequency,
                module=fm.module,
                source_id=fm.source_id,
            )
        )

    log.info(
        "lightcurvedb_export.built_cutouts",
        n=len(cutouts),
        n_without_thumbnail=len(flux_measurements) - len(cutouts),
    )
    return cutouts


async def _ingest(
    sources: list[Source],
    flux_measurements: list[FluxMeasurement],
    cutouts: list[Cutout],
    settings: LightcurveDBSettings,
):
    async with settings.backend as backend:
        await backend.sources.create_batch(sources)
        log.info("lightcurvedb_export.registered_sources", n=len(sources))

        await backend.fluxes.create_batch(flux_measurements)
        log.info("lightcurvedb_export.ingested_fluxes", n=len(flux_measurements))

        if cutouts:
            await backend.cutouts.create_batch(cutouts)
            log.info("lightcurvedb_export.ingested_cutouts", n=len(cutouts))


def main():
    parser = ap.ArgumentParser(
        description=__doc__,
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--discovery-parquet",
        type=str,
        required=True,
        help="Table produced by pickle_to_parquet.py.",
    )
    parser.add_argument(
        "--socat-db",
        type=str,
        default=None,
        help="socat sqlite db, for name -> socat_id linkage (optional).",
    )
    parser.add_argument(
        "--out-flux-parquet",
        type=str,
        default=None,
        help="Write the reshaped flux-measurement table here as well (cutouts, "
        "if any, are only ever pushed via --ingest, not written standalone).",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Actually push sources + fluxes through a lightcurvedb Backend "
        "(configured via LIGHTCURVEDB_* env vars, see lightcurvedb.config.Settings).",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.discovery_parquet)
    eligible = df[df["category"].isin(ELIGIBLE_CATEGORIES)]
    eligible = resolve_source_identity(eligible)
    log.info(
        "lightcurvedb_export.loaded",
        n_rows=len(df),
        n_eligible=len(eligible),
        by_category=eligible["category"].value_counts().to_dict(),
    )

    socat_name_to_id = (
        load_socat_name_to_id(Path(args.socat_db)) if args.socat_db else {}
    )
    sources, name_to_lc_id = build_lightcurvedb_sources(eligible, socat_name_to_id)
    flux_measurements = build_flux_measurements(eligible, name_to_lc_id)
    cutouts = build_cutouts(eligible, flux_measurements)

    if args.out_flux_parquet:
        frame = pd.DataFrame([fm.model_dump(mode="json") for fm in flux_measurements])
        frame.to_parquet(args.out_flux_parquet)
        log.info("lightcurvedb_export.wrote_flux_parquet", out=args.out_flux_parquet)

    if args.ingest:
        asyncio.run(
            _ingest(sources, flux_measurements, cutouts, LightcurveDBSettings())
        )


if __name__ == "__main__":
    main()
