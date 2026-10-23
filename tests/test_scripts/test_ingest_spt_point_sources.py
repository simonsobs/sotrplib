"""
Tests scripts/mock_spt_run/ingest_spt_point_sources.py: parsing SPT's
winter_combined.txt point-source list and selecting/flagging sources for
socat.

scripts/ is not a package (see scripts/mock_real_time/mockrt.py's
_load_script), so the module under test is loaded from its file path.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "mock_spt_run"
    / "ingest_spt_point_sources.py"
)


@pytest.fixture(scope="module")
def ingest_module():
    spec = importlib.util.spec_from_file_location(
        "ingest_spt_point_sources", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ─── load_rows ──────────────────────────────────────────────────────────────

SAMPLE = """\
# Main (combined) list including the following components.
# comment lines start with '#' and are skipped
"index"\t"ra(deg)"\t"dec(deg)"\t"sn90"\t"sn150"\t"sn220"\t"s90raw(mJy)"\t"s150raw(mJy)"\t"s220raw(mJy)"\t"class"
1\t10.0\t-50.0\t2.0\t1.0\t1.0\t350.0\t300.0\t250.0\t"thumb"
2\t11.0\t-51.0\t2.0\t1.0\t1.0\t50.0\t40.0\t30.0\t"ps"
3\t12.0\t-52.0\t2.0\t1.0\t1.0\t10.0\t8.0\t6.0\t"ps"
4\t13.0\t-53.0\t2.0\t1.0\t1.0\t-40.0\t-30.0\t-20.0\t"dec"
5\t14.0\t-54.0\t2.0\t1.0\t1.0\t400.0\t350.0\t300.0\t"edge"
"""


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "winter_combined.txt"
    path.write_text(SAMPLE)
    return path


def test_load_rows_skips_comments_and_header(ingest_module, sample_file):
    rows = ingest_module.load_rows(sample_file)
    assert len(rows) == 5
    assert rows[0] == {
        "index": 1,
        "ra_deg": 10.0,
        "dec_deg": -50.0,
        "s90raw_mjy": 350.0,
        "class": "thumb",
    }


def test_load_rows_reads_all_classes(ingest_module, sample_file):
    rows = ingest_module.load_rows(sample_file)
    assert {r["class"] for r in rows} == {"thumb", "ps", "dec", "edge"}


# ─── select_sources ─────────────────────────────────────────────────────────


def test_select_sources_default_classes_exclude_dec_and_edge(ingest_module, sample_file):
    rows = ingest_module.load_rows(sample_file)
    selected = ingest_module.select_sources(rows, monitored_min_mjy=20, pointing_min_mjy=300)
    assert {s["class"] for s in selected} == {"thumb", "ps"}
    assert {s["index"] for s in selected} == {1, 2}  # row 3 is below the 20mJy floor


def test_select_sources_flags_pointing_above_threshold(ingest_module, sample_file):
    rows = ingest_module.load_rows(sample_file)
    selected = ingest_module.select_sources(rows, monitored_min_mjy=20, pointing_min_mjy=300)
    by_index = {s["index"]: s for s in selected}
    assert by_index[1]["pointing"] is True  # 350 mJy > 300
    assert by_index[1]["monitored"] is True
    assert by_index[2]["pointing"] is False  # 50 mJy < 300
    assert by_index[2]["monitored"] is True


def test_select_sources_excludes_below_monitored_floor(ingest_module, sample_file):
    rows = ingest_module.load_rows(sample_file)
    selected = ingest_module.select_sources(rows, monitored_min_mjy=20, pointing_min_mjy=300)
    assert 3 not in {s["index"] for s in selected}  # 10 mJy <= 20 mJy floor


def test_select_sources_excludes_decrements_and_edge_by_default(ingest_module, sample_file):
    rows = ingest_module.load_rows(sample_file)
    selected = ingest_module.select_sources(rows, monitored_min_mjy=20, pointing_min_mjy=300)
    indices = {s["index"] for s in selected}
    assert 4 not in indices  # "dec", even though |flux| > floor
    assert 5 not in indices  # "edge", even though flux > both thresholds


def test_select_sources_name_uses_index(ingest_module, sample_file):
    rows = ingest_module.load_rows(sample_file)
    selected = ingest_module.select_sources(rows, monitored_min_mjy=20, pointing_min_mjy=300)
    assert {s["name"] for s in selected} == {"SPT-1", "SPT-2"}


def test_select_sources_classes_argument_overrides_default(ingest_module, sample_file):
    rows = ingest_module.load_rows(sample_file)
    selected = ingest_module.select_sources(
        rows, monitored_min_mjy=20, pointing_min_mjy=300, classes=("edge",)
    )
    assert {s["index"] for s in selected} == {5}
    assert selected[0]["pointing"] is True  # 400 mJy > 300
    assert selected[0]["monitored"] is True


# ─── ingest ─────────────────────────────────────────────────────────────────


def test_ingest_calls_create_source_with_flux_and_flags(ingest_module):
    client = MagicMock()
    sources = [
        {
            "index": 1,
            "ra_deg": 10.0,
            "dec_deg": -50.0,
            "s90raw_mjy": 350.0,
            "class": "thumb",
            "name": "SPT-1",
            "monitored": True,
            "pointing": True,
        }
    ]
    n = ingest_module.ingest(sources, client)
    assert n == 1
    client.create_source.assert_called_once()
    kwargs = client.create_source.call_args.kwargs
    assert kwargs["name"] == "SPT-1"
    assert kwargs["position"].ra.to_value("deg") == 10.0
    assert kwargs["position"].dec.to_value("deg") == -50.0
    assert kwargs["flux"].to_value("mJy") == 350.0
    assert kwargs["flags"] == {"monitored": True, "pointing": True}
