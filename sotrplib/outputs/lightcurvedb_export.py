"""
Convert a `pickle_to_parquet`-flattened table into lightcurvedb's ingest
schema, and (optionally) actually push it through `Backend.sources` /
`Backend.fluxes.ingest_dataframe` to prove the shape is correct.

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

lightcurvedb's `flux_measurements` table wants exactly:
    measurement_id (uuid, optional -- backends fill it in), frequency (int),
    module (str), source_id (uuid), time (timestamptz), ra, dec (deg),
    ra_uncertainty, dec_uncertainty (deg, nullable), flux, flux_err (Jy),
    extra (nullable json)
`source_id` must be a lightcurvedb `Source.source_id`, not sotrplib's own
`source_id` string (a catalog name like "ACT-S J0058.5+0620") -- so sources
have to be registered (or looked up) first and their name mapped to the
resulting UUID.
"""

import argparse as ap
import asyncio
import json
import sqlite3
from io import BytesIO
from pathlib import Path

import pandas as pd
from lightcurvedb.config import Settings as LightcurveDBSettings
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


def build_flux_measurement_frame(
    df: pd.DataFrame, name_to_lc_id: dict[str, str]
) -> pd.DataFrame:
    """Reshape an eligible-category slice of a `pickle_to_parquet` table
    (see `resolve_source_identity`) into lightcurvedb's flux-measurement
    ingest schema."""
    missing = set(df["_identity_name"]) - set(name_to_lc_id)
    if missing:
        raise ValueError(f"{len(missing)} source_id(s) have no lightcurvedb mapping")

    out = pd.DataFrame(
        {
            "source_id": df["_identity_name"].map(name_to_lc_id),
            "frequency": df["frequency"].round().astype(int),
            "module": df["array"],
            "time": pd.to_datetime(
                df["observation_mean_time_unix"], unit="s", utc=True
            ),
            "ra": df["ra"].astype(float),
            "dec": df["dec"].astype(float),
            "ra_uncertainty": df.get("err_ra"),
            "dec_uncertainty": df.get("err_dec"),
            "flux": df["flux"].astype(float) / 1000.0,  # mJy -> Jy
            "flux_err": df["err_flux"].astype(float) / 1000.0,  # mJy -> Jy
            "extra": None,
        }
    )
    return out


async def _ingest(
    sources: list[Source], flux_frame: pd.DataFrame, settings: LightcurveDBSettings
):
    async with settings.backend as backend:
        await backend.sources.create_batch(sources)
        log.info("lightcurvedb_export.registered_sources", n=len(sources))

        buffer = BytesIO()
        flux_frame.to_parquet(buffer)
        buffer.seek(0)
        await backend.fluxes.ingest_dataframe(buffer)
        log.info("lightcurvedb_export.ingested_fluxes", n=len(flux_frame))


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
        help="Write the reshaped flux-measurement table here instead of ingesting.",
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
    flux_frame = build_flux_measurement_frame(eligible, name_to_lc_id)

    if args.out_flux_parquet:
        flux_frame.to_parquet(args.out_flux_parquet)
        log.info("lightcurvedb_export.wrote_flux_parquet", out=args.out_flux_parquet)

    if args.ingest:
        asyncio.run(_ingest(sources, flux_frame, LightcurveDBSettings()))


if __name__ == "__main__":
    main()
