"""
Tests for the week planning and the configs of
scripts/coadding/submit_month_coadds.py.
"""

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sotrplib.config.coadd import CoaddSettings

SCRIPT_DIR = Path(__file__).parents[2] / "scripts" / "coadding"


@pytest.fixture(scope="module")
def month():
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        import submit_month_coadds

        yield submit_month_coadds
    finally:
        sys.path.remove(str(SCRIPT_DIR))


T0 = datetime(2025, 9, 4, 2, 0, tzinfo=timezone.utc)
DAY = timedelta(days=1)


def _fmt(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S.000000")


def _mapcat(path: Path, map_days: list[int], weeks: list[tuple[int, float]]):
    """Depth-1 f090 maps that start on map_days, and weekly coadds
    (start day, stop day)."""
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE depth_one_maps (map_name TEXT, frequency TEXT, "
        "start_time TEXT, stop_time TEXT)"
    )
    con.execute(
        "CREATE TABLE depth_one_coadds (coadd_name TEXT, coadd_type TEXT, "
        "frequency TEXT, start_time TEXT, stop_time TEXT)"
    )
    for d in map_days:
        t = T0 + d * DAY + timedelta(hours=1)
        con.execute(
            "INSERT INTO depth_one_maps VALUES (?, 'f090', ?, ?)",
            (f"m{d}", _fmt(t), _fmt(t + timedelta(hours=4))),
        )
    for start, stop in weeks:
        con.execute(
            "INSERT INTO depth_one_coadds VALUES (?, 'weekly', 'f090', ?, ?)",
            (
                f"w{start}",
                _fmt(T0 + start * DAY + timedelta(hours=1)),
                _fmt(T0 + stop * DAY),
            ),
        )
    con.commit()
    con.close()


def test_plan_weeks_uses_registered_weeks_and_makes_the_rest(tmp_path, month):
    db = tmp_path / "mapcat.sqlite"
    # Maps on all days 0..29; weeks 0 and 1 are registered.
    _mapcat(db, list(range(30)), [(0, 6.9), (7, 13.9)])
    start = T0.timestamp()
    stop = (T0 + 30 * DAY).timestamp()

    have, make = month.plan_weeks(db, "f090", "weekly", start, stop, 7.0)

    assert [round((s - start) / 86400) for s, _ in have] == [0, 7]
    assert [round((s - start) / 86400) for s, _ in make] == [14, 21, 28]
    # The last window is 2 days.
    assert round((make[-1][1] - make[-1][0]) / 86400) == 2


def test_plan_weeks_skips_windows_without_maps(tmp_path, month):
    db = tmp_path / "mapcat.sqlite"
    _mapcat(db, [0, 1, 15], [])
    start = T0.timestamp()
    have, make = month.plan_weeks(
        db, "f090", "weekly", start, (T0 + 28 * DAY).timestamp(), 7.0
    )
    assert have == []
    assert [round((s - start) / 86400) for s, _ in make] == [0, 14]


def test_plan_weeks_rejects_a_registered_week_that_crosses_the_window(tmp_path, month):
    db = tmp_path / "mapcat.sqlite"
    # A registered "week" from day 28 to day 33, in a month of 30 days.
    _mapcat(db, list(range(30)), [(28, 33)])
    with pytest.raises(ValueError, match="stops at"):
        month.plan_weeks(
            db,
            "f090",
            "weekly",
            T0.timestamp(),
            (T0 + 30 * DAY).timestamp(),
            7.0,
        )


def test_main_writes_valid_configs_and_dependencies(tmp_path, month):
    db = tmp_path / "mapcat.sqlite"
    _mapcat(db, list(range(28)), [(0, 6.9)])
    submit = month.main(
        [
            "--database-name",
            str(db),
            "--depth-one-parent",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "monthly"),
            "--start-time",
            T0.isoformat(),
            "--month-days",
            "28",
            "--frequencies",
            "f090",
            "--ephem-file-path",
            "",
        ]
    )
    slurm_dir = submit.parent
    weeks = sorted(slurm_dir.glob("week_*.json"))
    assert [w.name for w in weeks] == [
        "week_20250911_f090.json",
        "week_20250918_f090.json",
        "week_20250925_f090.json",
    ]
    for w in weeks:
        config = CoaddSettings.from_file(w)
        assert not config.inputs_are_coadds
        assert config.mapcat_registration.coadd_type == "weekly"

    (month_file,) = slurm_dir.glob("month_*.json")
    config = CoaddSettings.from_file(month_file)
    assert config.inputs_are_coadds
    assert config.maps.coadd_type == "weekly"
    assert config.maps.time_binning == "left-bound"
    assert config.preprocessors == []
    assert config.mapcat_registration.coadd_type == "monthly"
    assert json.loads(month_file.read_text())["maps"]["map_type"] == "coadd_rhokappa"

    text = submit.read_text()
    assert text.count("sbatch --parsable week_") == 3
    assert "--dependency=afterok" in text
    assert text.strip().endswith("month_20250904_f090.slurm")


def test_main_no_week_coadd_makes_one_depth_one_job(tmp_path, month):
    db = tmp_path / "mapcat.sqlite"
    _mapcat(db, list(range(28)), [(0, 6.9)])
    submit = month.main(
        [
            "--database-name",
            str(db),
            "--depth-one-parent",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "monthly"),
            "--start-time",
            T0.isoformat(),
            "--frequencies",
            "f090",
            "--no-week-coadd",
            "--ephem-file-path",
            "",
        ]
    )
    slurm_dir = submit.parent
    assert list(slurm_dir.glob("week_*")) == []
    (month_file,) = slurm_dir.glob("month_*.json")
    config = CoaddSettings.from_file(month_file)
    assert not config.inputs_are_coadds
    assert config.mapcat_registration.coadd_type == "monthly"
    assert "dependency" not in submit.read_text()
