"""
Convert `PickleSerializer` output (see `sotrplib.outputs.core`) into a single
tabular Parquet file.

Each pickle holds one map's worth of `MeasuredSource` lists nested under
different keys (`forced_photometry`, `sifted_blind_search.source_candidates`,
`sifted_blind_search.transient_candidates`, `sifted_blind_search.noise_candidates`,
`pointing_sources`). This flattens all of those, across as many pickles as
you like, into one row-per-source table tagged with `map_id`/`category`/
`source_pickle`, dropping the per-source `thumbnail` array by default since
that's what makes the raw pickles so large (see `--keep-thumbnails` to keep
it as a JSON-encoded column instead).

Quantity fields (ra, flux, ...) are stored as plain floats in the unit fixed
by the field's pydantic annotation (deg, mJy, GHz, ...); list/dict-valued
fields (crossmatches, flags, alternate_names, fit_params) are JSON-encoded
strings so every row shares a flat, Parquet-friendly schema.
"""

import argparse as ap
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from pydantic import BaseModel
from structlog import get_logger

log = get_logger()

CATEGORIES = {
    "forced_photometry": lambda d: d.get("forced_photometry") or [],
    "pointing_sources": lambda d: d.get("pointing_sources") or [],
    "source_candidates": lambda d: (
        getattr(d.get("sifted_blind_search"), "source_candidates", None) or []
    ),
    "transient_candidates": lambda d: (
        getattr(d.get("sifted_blind_search"), "transient_candidates", None) or []
    ),
    "noise_candidates": lambda d: (
        getattr(d.get("sifted_blind_search"), "noise_candidates", None) or []
    ),
}

# Fixed unit each field is stored in, so a column stays consistent across
# rows regardless of which Quantity subclass (Longitude, AstroPydanticQuantity,
# plain Quantity, ...) the pipeline happened to leave in that attribute at
# pickle time -- real MeasuredSource objects mix these depending on how the
# value was set, so we can't rely on pydantic's own JSON serialization.
FIELD_UNITS = {
    "ra": u.deg,
    "dec": u.deg,
    "err_ra": u.deg,
    "err_dec": u.deg,
    "offset_ra": u.deg,
    "offset_dec": u.deg,
    "fwhm_ra": u.deg,
    "fwhm_dec": u.deg,
    "err_fwhm_ra": u.deg,
    "err_fwhm_dec": u.deg,
    "flux": u.mJy,
    "err_flux": u.mJy,
    "frequency": u.GHz,
    "thumbnail_res": u.arcmin,
    "angular_separation": u.deg,
    "distance": u.Mpc,
}


def _plain_scalar(name: str, value):
    """
    Reduce a single attribute value to something Parquet can store, or a
    sentinel dict `{"iso": ..., "unix": ...}` for Time values (expanded into
    two columns by the caller).
    """
    if value is None:
        return None
    if isinstance(value, Time):
        return {"iso": value.isot[:23], "unix": float(value.unix)}
    if isinstance(value, u.Quantity):
        target = FIELD_UNITS.get(name)
        try:
            v = value.to_value(target) if target is not None else value.value
        except u.UnitConversionError:
            v = value.value
        arr = np.atleast_1d(v)
        # thumbnail_res etc. carry a (ra, dec) pixel-scale pair rather than
        # a single scalar -- fall back to a plain list for those.
        return float(arr.item()) if arr.size == 1 else [float(x) for x in arr]
    if isinstance(value, u.UnitBase):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _json_default(obj):
    """Fallback encoder for `json.dumps` covering leftover numpy/Quantity
    values nested inside free-form dicts like `fit_params`. Quantity is
    itself an ndarray subclass (and overrides `.tolist()` to raise), so it
    must be checked before the plain-ndarray case."""
    if isinstance(obj, u.Quantity):
        return _plain_scalar("", obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def _crossmatch_to_dict(crossmatch: BaseModel) -> dict:
    d = {}
    for name in type(crossmatch).model_fields:
        plain = _plain_scalar(name, getattr(crossmatch, name, None))
        if isinstance(plain, dict) and "unix" in plain:
            d[f"{name}_iso"] = plain["iso"]
            d[f"{name}_unix"] = plain["unix"]
        else:
            d[name] = plain
    return d


def flatten_source(source: BaseModel, keep_thumbnails: bool = False) -> dict:
    """
    Flatten a single `MeasuredSource` (or other pydantic source model) into
    a dict of Parquet-friendly scalars, keyed off its *declared* pydantic
    fields (not `model_dump`, which chokes on the mixed Quantity subclasses
    real pipeline objects end up holding).
    """
    flat = {}
    for name in type(source).model_fields:
        if name == "thumbnail":
            value = getattr(source, "thumbnail", None)
            if keep_thumbnails and value is not None:
                flat["thumbnail"] = json.dumps(np.asarray(value).tolist())
            continue
        if name == "crossmatches":
            crossmatches = getattr(source, "crossmatches", None) or []
            flat["crossmatches"] = (
                json.dumps(
                    [_crossmatch_to_dict(cm) for cm in crossmatches],
                    default=_json_default,
                )
                if crossmatches
                else None
            )
            flat["n_crossmatches"] = len(crossmatches)
            continue

        plain = _plain_scalar(name, getattr(source, name, None))
        if isinstance(plain, dict) and "unix" in plain:
            flat[f"{name}_iso"] = plain["iso"]
            flat[f"{name}_unix"] = plain["unix"]
        elif isinstance(plain, (list, dict)):
            flat[name] = json.dumps(plain, default=_json_default) if plain else None
        else:
            flat[name] = plain
    return flat


def records_from_pickle(pickle_path: Path, keep_thumbnails: bool = False) -> list[dict]:
    """
    Load one `PickleSerializer` pickle and return a flat list of row-dicts,
    one per source across every category, tagged with map_id/category/
    source_pickle.
    """
    with open(pickle_path, "rb") as handle:
        d = pickle.load(handle)

    map_id = d.get("map_id", "")
    records = []
    for category, getter in CATEGORIES.items():
        sources = getter(d)
        for source in sources:
            if not isinstance(source, BaseModel):
                log.warning(
                    "pickle_to_parquet.skipping_non_pydantic_source",
                    pickle=str(pickle_path),
                    category=category,
                    type=type(source).__name__,
                )
                continue
            record = flatten_source(source, keep_thumbnails=keep_thumbnails)
            record["map_id"] = map_id
            record["category"] = category
            record["source_pickle"] = pickle_path.name
            records.append(record)
    return records


def convert(
    pickle_paths: list[Path],
    out_path: Path,
    keep_thumbnails: bool = False,
) -> pd.DataFrame:
    """
    Flatten every pickle in `pickle_paths` into one combined dataframe and
    write it to `out_path` as Parquet. Returns the dataframe.
    """
    records = []
    for pickle_path in pickle_paths:
        try:
            records.extend(
                records_from_pickle(pickle_path, keep_thumbnails=keep_thumbnails)
            )
        except Exception as exc:
            log.error(
                "pickle_to_parquet.unreadable_pickle",
                pickle=str(pickle_path),
                error=str(exc),
            )

    df = pd.DataFrame.from_records(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info(
        "pickle_to_parquet.wrote_parquet",
        out=str(out_path),
        n_pickles=len(pickle_paths),
        n_rows=len(df),
    )
    return df


def main():
    parser = ap.ArgumentParser(
        description=__doc__,
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pickle-dir",
        type=str,
        default=None,
        help="Directory to glob for pickles (see --pattern). Ignored if --files is set.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pickle",
        help="Glob pattern used with --pickle-dir.",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of pickle files to convert, instead of --pickle-dir.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output Parquet file path.",
    )
    parser.add_argument(
        "--keep-thumbnails",
        action="store_true",
        help="Keep each source's thumbnail array as a JSON-encoded column "
        "(large; defeats the point of converting away from pickle if the "
        "source pickles carried many-source thumbnails).",
    )
    args = parser.parse_args()

    if args.files:
        pickle_paths = [Path(f) for f in args.files]
    elif args.pickle_dir:
        pickle_paths = sorted(Path(args.pickle_dir).glob(args.pattern))
    else:
        parser.error("one of --files or --pickle-dir is required")

    if not pickle_paths:
        parser.error("no pickle files found")

    convert(pickle_paths, Path(args.out), keep_thumbnails=args.keep_thumbnails)


if __name__ == "__main__":
    main()
