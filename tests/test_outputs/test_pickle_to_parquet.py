"""
Tests for pickle_to_parquet, in particular that flattening survives the
mixed Quantity subclasses (Longitude, AstroPydanticQuantity, plain
Quantity, ...) real pipeline objects hold at runtime -- model_dump(mode=
"json") chokes on those, so flattening is done off the declared pydantic
fields with duck-typed conversion instead. See the module docstring.
"""

import json

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time

from sotrplib.outputs.core import PickleSerializer
from sotrplib.outputs.pickle_to_parquet import convert
from sotrplib.sifter.core import SifterResult
from sotrplib.sources.sources import CrossMatch, MeasuredSource


def make_candidate(**overrides):
    defaults = dict(
        ra=10.0 * u.deg,
        dec=5.0 * u.deg,
        flux=42.0 * u.mJy,
        err_flux=1.0 * u.mJy,
        frequency=90.0 * u.GHz,
        array="i1",
        snr=6.5,
        source_id="SO-SV_J000000.0+000000_001",
        observation_mean_time=Time("2025-09-10T00:00:00"),
        fit_params={"amplitude": 1.2, "cov": np.eye(2)},
        crossmatches=[CrossMatch(source_id="ACT-S J0000.0+0000", catalog_name="socat")],
        thumbnail=np.zeros((4, 4)),
        thumbnail_res=np.array([0.5, 0.5]) * u.arcmin,
    )
    defaults.update(overrides)
    return MeasuredSource(**defaults)


def test_convert_flattens_forced_photometry(tmp_path):
    candidate = make_candidate()
    pickle_path = list(_write_pickle(tmp_path, forced_photometry=[candidate]))[0]

    out_parquet = tmp_path / "out.parquet"
    df = convert([pickle_path], out_parquet)

    assert out_parquet.exists()
    assert len(df) == 1
    row = df.iloc[0]

    assert row["source_id"] == "SO-SV_J000000.0+000000_001"
    assert row["ra"] == 10.0
    assert row["dec"] == 5.0
    assert row["flux"] == 42.0
    assert row["frequency"] == 90.0
    assert row["category"] == "forced_photometry"
    assert row["map_id"] == "test_map"
    assert row["n_crossmatches"] == 1
    assert "thumbnail" not in df.columns  # dropped by default

    crossmatches = json.loads(row["crossmatches"])
    assert crossmatches[0]["source_id"] == "ACT-S J0000.0+0000"

    fit_params = json.loads(row["fit_params"])
    assert fit_params["amplitude"] == 1.2
    assert fit_params["cov"] == [[1.0, 0.0], [0.0, 1.0]]

    # a 2-element Quantity field falls back to a JSON list, not a float
    assert json.loads(row["thumbnail_res"]) == [0.5, 0.5]


def test_convert_reads_from_parquet_roundtrip(tmp_path):
    candidate = make_candidate(source_id="SO-SV_J111111.1+111111_001")
    pickle_path = list(_write_pickle(tmp_path, forced_photometry=[candidate]))[0]
    out_parquet = tmp_path / "out.parquet"
    convert([pickle_path], out_parquet)

    reloaded = pd.read_parquet(out_parquet)
    assert reloaded.iloc[0]["source_id"] == "SO-SV_J111111.1+111111_001"


def test_convert_covers_sifter_categories(tmp_path):
    transient = make_candidate(source_id="transient-1")
    noise = make_candidate(source_id="noise-1")
    pickle_path = list(
        _write_pickle(
            tmp_path,
            forced_photometry=[],
            sifter_result=SifterResult(
                source_candidates=[],
                transient_candidates=[transient],
                noise_candidates=[noise],
            ),
        )
    )[0]

    df = convert([pickle_path], tmp_path / "out.parquet")
    assert set(df["category"]) == {"transient_candidates", "noise_candidates"}


def test_keep_thumbnails_flag(tmp_path):
    candidate = make_candidate()
    pickle_path = list(_write_pickle(tmp_path, forced_photometry=[candidate]))[0]

    df = convert([pickle_path], tmp_path / "out.parquet", keep_thumbnails=True)
    assert "thumbnail" in df.columns
    thumb = json.loads(df.iloc[0]["thumbnail"])
    assert np.array(thumb).shape == (4, 4)


def _write_pickle(
    tmp_path,
    forced_photometry=None,
    sifter_result=None,
    map_id="test_map",
):
    serializer = PickleSerializer(directory=tmp_path)
    serializer.output(
        forced_photometry_candidates=forced_photometry or [],
        sifter_result=sifter_result
        or SifterResult(
            source_candidates=[], transient_candidates=[], noise_candidates=[]
        ),
        map_id=map_id,
    )
    return sorted(tmp_path.glob("*.pickle"))
