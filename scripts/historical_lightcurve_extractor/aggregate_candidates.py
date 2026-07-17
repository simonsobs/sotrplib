"""
Aggregate blind-search candidates across many sifter-output pickles into a
reviewable, deterministic list of "worth a historical lightcurve" targets.

Automates what a human currently does by hand in a notebook: pool
transient_candidates (optionally noise_candidates too) across every pickle
in a run, drop anything already matched to a known catalog source, apply a
few QA cuts, then cluster by sky position (and, for depth-1 pickles, time)
to merge repeat detections of the same object across bands/tubes/epochs
into one candidate. Repeat-confirmed candidates (>=2 detections) go to
--out-csv, which dispatch_historical_extraction.py (Script B) consumes
directly -- no human review step, per an explicit choice to run this fully
automated. Candidates with only a single detection go to
--singleton-out-csv instead: not auto-dispatched, kept for manual
follow-up.

Works unmodified against both today's week-coadd pickles (map_id is a bare
mapcat coadd UUID, no shared observation epoch) and a future full depth-1
run (map_id is "{band}_{array}_{unix}", so records additionally get
time-bucketed before clustering) -- just point --pickle-dir at whichever
output and retune the QA/cluster flags for that run's candidate volume.
"""

import argparse
import pickle as pickle_module
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import structlog
from astropy.time import Time
from sklearn.cluster import DBSCAN

from sotrplib.sources.sources import MeasuredSource
from sotrplib.utils.utils import get_fwhm, radec_to_str_name

log = structlog.get_logger()

_MAP_ID_RE = re.compile(r"^(f\d{3})_([A-Za-z0-9]+)_(\d+)$")
_BAND_GHZ = {"f030": 30, "f040": 40, "f090": 90, "f150": 150, "f220": 220, "f280": 280}
_BAND_GHZ_TOLERANCE = 40.0


@dataclass
class CandidateRecord:
    source: MeasuredSource
    map_id: str
    band: str | None
    tube: str | None
    epoch_unix: int | None
    bucket: Literal["transient", "noise"]
    pickle_path: Path
    qa_flags: list[str] = field(default_factory=list)


def find_pickle_files(root: Path, pattern: str, recursive: bool) -> list[Path]:
    return sorted(root.rglob(pattern) if recursive else root.glob(pattern))


def parse_map_id(map_id: str) -> tuple[str | None, str | None, int | None]:
    """Depth-1 map_ids look like "f090_i1_1725840000"
    (ProcessableMap.get_map_str_id's default). Coadd map_ids are a bare
    mapcat UUID with no embedded band/tube/time -- don't guess at those,
    return all-None so callers fall back to other sources of band/tube
    info instead of silently mis-parsing a UUID's first two "_"-split
    tokens.
    """
    match = _MAP_ID_RE.match(map_id)
    if not match:
        return None, None, None
    band, tube, unix_str = match.groups()
    return band, tube, int(unix_str)


def band_tag_from_ghz(freq_ghz: float | None) -> str | None:
    if freq_ghz is None:
        return None
    nearest_tag, nearest_diff = None, None
    for tag, ghz in _BAND_GHZ.items():
        diff = abs(ghz - freq_ghz)
        if nearest_diff is None or diff < nearest_diff:
            nearest_tag, nearest_diff = tag, diff
    if nearest_diff is None or nearest_diff > _BAND_GHZ_TOLERANCE:
        return None
    return nearest_tag


def load_candidates(
    pickle_paths: list[Path],
    include_noise: bool,
) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for pickle_path in pickle_paths:
        try:
            with open(pickle_path, "rb") as fh:
                data = pickle_module.load(fh)
        except Exception:
            log.warning(
                "aggregate_candidates.load_candidates.unreadable_pickle",
                pickle_path=str(pickle_path),
                exc_info=True,
            )
            continue

        map_id = str(data.get("map_id", pickle_path.stem))
        band, tube, epoch_unix = parse_map_id(map_id)

        sifted = data.get("sifted_blind_search")
        if sifted is None:
            continue

        buckets: list[tuple[list[MeasuredSource], Literal["transient", "noise"]]] = [
            (sifted.transient_candidates, "transient"),
        ]
        if include_noise:
            buckets.append((sifted.noise_candidates, "noise"))

        for sources, bucket in buckets:
            for source in sources:
                resolved_band = band
                resolved_tube = tube
                if resolved_band is None:
                    resolved_band = band_tag_from_ghz(
                        source.frequency.to_value("GHz")
                        if source.frequency is not None
                        else None
                    )
                if resolved_tube is None:
                    resolved_tube = source.array

                records.append(
                    CandidateRecord(
                        source=source,
                        map_id=map_id,
                        band=resolved_band,
                        tube=resolved_tube,
                        epoch_unix=epoch_unix,
                        bucket=bucket,
                        pickle_path=pickle_path,
                    )
                )

    log.info(
        "aggregate_candidates.load_candidates.complete",
        n_pickles=len(pickle_paths),
        n_records=len(records),
    )
    return records


def apply_time_window(
    records: list[CandidateRecord],
    start_unix: float | None,
    stop_unix: float | None,
) -> list[CandidateRecord]:
    """Only filters records with a resolvable epoch_unix (depth-1 pickles).
    Coadd/UUID-keyed records have no single shared observation epoch to
    window on, so they're kept unconditionally rather than being silently
    dropped by a filter that can't apply to them."""
    if start_unix is None and stop_unix is None:
        return records
    kept = []
    for record in records:
        if record.epoch_unix is None:
            kept.append(record)
            continue
        if start_unix is not None and record.epoch_unix < start_unix:
            continue
        if stop_unix is not None and record.epoch_unix > stop_unix:
            continue
        kept.append(record)
    return kept


def exclude_known_catalog_matches(
    records: list[CandidateRecord],
    include_known_source_flares: bool,
) -> tuple[list[CandidateRecord], list[CandidateRecord]]:
    """A non-empty .crossmatches on a transient/noise candidate means
    "known catalog source, currently flaring or cut" (see sift()), not
    "solar system object" or "genuinely unidentified" -- source_type is
    not populated by sift() itself and can't be used for this. Returns
    (kept, excluded); excluded is only non-empty for inspection/logging
    when include_known_source_flares is False.
    """
    kept, excluded = [], []
    for record in records:
        if record.source.crossmatches:
            excluded.append(record)
            if include_known_source_flares:
                kept.append(record)
        else:
            kept.append(record)
    return kept, excluded


def apply_qa_cuts(
    records: list[CandidateRecord],
    min_snr: float,
    max_fwhm_ratio: float,
    max_flux_jy: float,
    require_thumbnail: bool,
) -> tuple[list[CandidateRecord], dict[str, int]]:
    tally: Counter[str] = Counter()
    kept = []
    for record in records:
        source = record.source

        if require_thumbnail and source.thumbnail is None:
            tally["no_thumbnail"] += 1
            continue

        if source.snr is None or source.snr < min_snr:
            tally["min_snr"] += 1
            continue

        if source.flux is None or not np.isfinite(source.flux.to_value("Jy")):
            tally["non_finite_flux"] += 1
            continue
        flux_jy = source.flux.to_value("Jy")
        if flux_jy <= 0 or flux_jy > max_flux_jy:
            tally["max_flux_jy"] += 1
            continue

        if record.band is not None and (
            source.fwhm_ra is not None and source.fwhm_dec is not None
        ):
            try:
                nominal_fwhm = get_fwhm(record.band, record.tube).to_value("arcmin")
            except KeyError:
                nominal_fwhm = None
            if nominal_fwhm is not None:
                fwhm_ra = source.fwhm_ra.to_value("arcmin")
                fwhm_dec = source.fwhm_dec.to_value("arcmin")
                if (
                    fwhm_ra > max_fwhm_ratio * nominal_fwhm
                    or fwhm_dec > max_fwhm_ratio * nominal_fwhm
                ):
                    tally["max_fwhm_ratio"] += 1
                    continue

        if record.band is None:
            record.qa_flags.append("unresolved_band")
        if record.tube is None:
            record.qa_flags.append("unresolved_tube")
        if record.epoch_unix is None:
            record.qa_flags.append("coadd_no_epoch")

        kept.append(record)

    log.info("aggregate_candidates.apply_qa_cuts.complete", **dict(tally))
    return kept, dict(tally)


def cluster_candidates(
    records: list[CandidateRecord],
    eps_arcmin: float,
    eps_hours: float,
) -> list[list[CandidateRecord]]:
    """Merge repeat detections of the same object across bands/tubes/
    pickles by sky position. depth-1 records (resolvable epoch_unix) are
    additionally time-bucketed before spatial clustering -- a fixed time
    window is meaningless for coadd records, since a coadd candidate's own
    observation_mean_time is an average over whichever depth-1 visits
    happened to touch that pixel, not a real shared observation epoch."""
    depth1 = [r for r in records if r.epoch_unix is not None]
    coadd = [r for r in records if r.epoch_unix is None]

    clusters: list[list[CandidateRecord]] = []
    clusters.extend(_spatial_cluster(coadd, eps_arcmin))

    if depth1:
        bucket_width_sec = eps_hours * 3600.0
        time_buckets: dict[int, list[CandidateRecord]] = {}
        for record in depth1:
            time_buckets.setdefault(
                int(record.epoch_unix // bucket_width_sec), []
            ).append(record)
        for bucket_records in time_buckets.values():
            clusters.extend(_spatial_cluster(bucket_records, eps_arcmin))

    return clusters


def _spatial_cluster(
    records: list[CandidateRecord], eps_arcmin: float
) -> list[list[CandidateRecord]]:
    if not records:
        return []
    if len(records) == 1:
        return [records]

    dec_rad = np.array([r.source.dec.to_value("rad") for r in records])
    ra_rad = np.array([r.source.ra.to_value("rad") for r in records])
    mean_dec = np.mean(dec_rad)
    # Flat-sky approximation (arcmin-scale eps): x = ra*cos(dec), y = dec.
    x = np.degrees(ra_rad) * 60.0 * np.cos(mean_dec)
    y = np.degrees(dec_rad) * 60.0
    X = np.vstack([x, y]).T

    labels = DBSCAN(eps=eps_arcmin, min_samples=1, metric="euclidean").fit_predict(X)

    clusters: dict[int, list[CandidateRecord]] = {}
    for label, record in zip(labels, records):
        clusters.setdefault(int(label), []).append(record)
    return list(clusters.values())


def summarize_cluster(cluster: list[CandidateRecord], used_names: Counter) -> dict:
    representative = max(cluster, key=lambda r: r.source.snr)

    ras = np.array([r.source.ra.to_value("deg") for r in cluster])
    decs = np.array([r.source.dec.to_value("deg") for r in cluster])
    mean_ra, mean_dec = float(np.mean(ras)), float(np.mean(decs))
    if len(cluster) > 1:
        seps = np.sqrt(
            ((ras - ras[:, None]) * np.cos(np.radians(mean_dec))) ** 2
            + (decs - decs[:, None]) ** 2
        )
        position_scatter_arcmin = float(np.max(seps) * 60.0)
    else:
        position_scatter_arcmin = 0.0

    base_name = radec_to_str_name(
        representative.source.ra.to_value("deg"),
        representative.source.dec.to_value("deg"),
        source_class="transient",
    ).replace(" ", "_")
    used_names[base_name] += 1
    cluster_id = f"{base_name}_{used_names[base_name]:03d}"

    bands = sorted({r.band for r in cluster if r.band is not None})
    tubes = sorted({r.tube for r in cluster if r.tube is not None})
    epochs = [r.epoch_unix for r in cluster if r.epoch_unix is not None]
    qa_flags = sorted({flag for r in cluster for flag in r.qa_flags})
    if len(cluster) == 1:
        qa_flags = ["singleton"] + qa_flags

    crossmatched = any(r.source.crossmatches for r in cluster)
    known_source_id = ""
    if crossmatched:
        for r in cluster:
            if r.source.crossmatches:
                known_source_id = str(r.source.crossmatches[0].source_id)
                break

    flux_mjy = representative.source.flux.to_value("mJy")

    return {
        "cluster_id": cluster_id,
        "ra_deg": mean_ra,
        "dec_deg": mean_dec,
        "position_scatter_arcmin": position_scatter_arcmin,
        "flux_mjy": flux_mjy,
        "flux_jy": flux_mjy / 1000.0,
        "max_snr": float(representative.source.snr),
        "n_detections": len(cluster),
        "n_bands": len(bands),
        "n_tubes": len(tubes),
        "bands": ";".join(bands),
        "tubes": ";".join(tubes),
        "map_ids": ";".join(sorted({r.map_id for r in cluster})),
        "first_obs_unix": min(epochs) if epochs else "",
        "last_obs_unix": max(epochs) if epochs else "",
        "fwhm_ra_arcmin": (
            representative.source.fwhm_ra.to_value("arcmin")
            if representative.source.fwhm_ra is not None
            else ""
        ),
        "fwhm_dec_arcmin": (
            representative.source.fwhm_dec.to_value("arcmin")
            if representative.source.fwhm_dec is not None
            else ""
        ),
        "crossmatched": crossmatched,
        "known_source_id": known_source_id,
        "thumbnail_png": "",
        "qa_flags": ";".join(qa_flags),
        "_representative": representative,
    }


def save_thumbnail_png(row: dict, out_dir: Path) -> str:
    representative: CandidateRecord = row["_representative"]
    thumbnail = representative.source.thumbnail
    if thumbnail is None or np.all(np.isnan(thumbnail)):
        return ""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{row['cluster_id']}.png"
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(thumbnail, origin="lower", cmap="inferno")
    ax.set_title(row["cluster_id"])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(png_path, dpi=100)
    plt.close(fig)
    return str(png_path)


_CSV_COLUMNS = [
    "cluster_id",
    "ra_deg",
    "dec_deg",
    "position_scatter_arcmin",
    "flux_mjy",
    "flux_jy",
    "max_snr",
    "n_detections",
    "n_bands",
    "n_tubes",
    "bands",
    "tubes",
    "map_ids",
    "first_obs_unix",
    "last_obs_unix",
    "fwhm_ra_arcmin",
    "fwhm_dec_arcmin",
    "crossmatched",
    "known_source_id",
    "thumbnail_png",
    "qa_flags",
]


def write_csv(rows: list[dict], out_csv: Path) -> None:
    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in _CSV_COLUMNS})


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        prog="aggregate-candidates",
        description=(
            "Aggregate blind-search candidates across sifter-output pickles "
            "into repeat-confirmed (candidates.csv) and singleton "
            "(singleton-out-csv) target lists for historical lightcurve "
            "dispatch."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pickle-dir", type=Path, required=True)
    parser.add_argument("--glob-pattern", type=str, default="*.pickle")
    parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./"))
    parser.add_argument("--out-csv", type=str, default="candidates.csv")
    parser.add_argument("--singleton-out-csv", type=str, default=None)
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="ISO8601. Only filters depth-1 records with a resolvable "
        "epoch_unix; coadd/UUID-keyed records are always kept.",
    )
    parser.add_argument("--stop-time", type=str, default=None)
    parser.add_argument(
        "--include-noise-candidates", action="store_true", default=False
    )
    parser.add_argument("--min-snr", type=float, default=8.0)
    parser.add_argument("--max-fwhm-ratio", type=float, default=3.0)
    parser.add_argument("--max-flux-jy", type=float, default=100.0)
    parser.add_argument("--require-thumbnail", action="store_true", default=False)
    parser.add_argument(
        "--include-known-source-flares", action="store_true", default=False
    )
    parser.add_argument("--cluster-eps-arcmin", type=float, default=5.0)
    parser.add_argument("--cluster-eps-hours", type=float, default=6.0)
    parser.add_argument(
        "--save-thumbnail-pngs", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    out_csv = args.out_dir / args.out_csv
    singleton_out_csv = args.out_dir / (
        args.singleton_out_csv or f"{Path(args.out_csv).stem}_singletons.csv"
    )

    start_unix = Time(args.start_time).unix if args.start_time else None
    stop_unix = Time(args.stop_time).unix if args.stop_time else None

    pickle_paths = find_pickle_files(args.pickle_dir, args.glob_pattern, args.recursive)
    log.info("aggregate_candidates.main.found_pickles", n=len(pickle_paths))

    records = load_candidates(pickle_paths, include_noise=args.include_noise_candidates)
    records = apply_time_window(records, start_unix, stop_unix)
    records, _excluded = exclude_known_catalog_matches(
        records, include_known_source_flares=args.include_known_source_flares
    )
    records, _tally = apply_qa_cuts(
        records,
        min_snr=args.min_snr,
        max_fwhm_ratio=args.max_fwhm_ratio,
        max_flux_jy=args.max_flux_jy,
        require_thumbnail=args.require_thumbnail,
    )

    clusters = cluster_candidates(
        records,
        eps_arcmin=args.cluster_eps_arcmin,
        eps_hours=args.cluster_eps_hours,
    )
    log.info("aggregate_candidates.main.clustered", n_clusters=len(clusters))

    used_names: Counter = Counter()
    repeat_rows, singleton_rows = [], []
    for cluster in clusters:
        row = summarize_cluster(cluster, used_names)
        if args.save_thumbnail_pngs:
            row["thumbnail_png"] = save_thumbnail_png(row, args.out_dir / "thumbnails")
        del row["_representative"]
        (repeat_rows if row["n_detections"] >= 2 else singleton_rows).append(row)

    write_csv(repeat_rows, out_csv)
    write_csv(singleton_rows, singleton_out_csv)

    log.info(
        "aggregate_candidates.main.complete",
        n_repeat_confirmed=len(repeat_rows),
        n_singleton=len(singleton_rows),
        out_csv=str(out_csv),
        singleton_out_csv=str(singleton_out_csv),
    )
    print(f"Repeat-confirmed candidates ({len(repeat_rows)}): {out_csv}")
    print(f"Singleton candidates ({len(singleton_rows)}): {singleton_out_csv}")


if __name__ == "__main__":  # pragma: no cover
    main()
