"""
Render the regression summary as a markdown report, optionally comparing
against a baseline summary (e.g., from the latest run on main).

Usage:
    python report.py summary.json [--baseline baseline.json]
        [--baseline-status STATUS] [--baseline-url URL] [--run-url URL]
"""

import argparse
import json
from pathlib import Path

# Hidden marker used to find and update an existing PR comment.
MARKER = "<!-- sotrplib-regression-report -->"

# (key, label, kind) grouped by section. kind is "time", "count", or "float".
SECTIONS = {
    "Performance": [
        ("duration", "Runtime (s)", "time"),
    ],
    "Sources": [
        ("number_with_forced_photometry", "Forced photometry", "count"),
        ("number_forced_with_free_fit", "Forced with free fit", "count"),
        ("number_with_pointing_fit", "Pointing fits", "count"),
        ("number_blind_sources", "Blind sources", "count"),
        ("number_noise_sources", "Noise sources", "count"),
        ("number_transient_sources", "Transient sources", "count"),
    ],
    "Recovered source statistics": [
        ("mean_offset_deg", "Mean offset (deg)", "float"),
        ("median_offset_deg", "Median offset (deg)", "float"),
        ("std_offset_deg", "Std offset (deg)", "float"),
        ("mean_ra_offset_deg", "Mean RA offset (deg)", "float"),
        ("median_ra_offset_deg", "Median RA offset (deg)", "float"),
        ("std_ra_offset_deg", "Std RA offset (deg)", "float"),
        ("mean_dec_offset_deg", "Mean Dec offset (deg)", "float"),
        ("median_dec_offset_deg", "Median Dec offset (deg)", "float"),
        ("std_dec_offset_deg", "Std Dec offset (deg)", "float"),
        ("mean_flux_ratio", "Mean flux ratio", "float"),
        ("median_flux_ratio", "Median flux ratio", "float"),
        ("std_flux_ratio", "Std flux ratio", "float"),
        ("median_flux_ratio_above_100mJy", "Median flux ratio (>100 mJy)", "float"),
    ],
}

# Relative change above which a metric is flagged. Runtime on shared CI
# runners is noisy, so it gets a looser threshold.
THRESHOLDS = {"time": 0.2, "count": 0.0, "float": 0.1}

CHANGE_ICON = "⚠️"
MISSING_ICON = "❓"


def fmt(value, kind):
    if value is None:
        return "–"
    if kind == "count":
        return f"{int(value)}"
    if kind == "time":
        return f"{value:.1f}"
    return f"{value:.4g}"


def fmt_delta(new, old, kind):
    """Return (delta string, flag icon or "")."""
    if new is None or old is None:
        # Metric present on only one side, e.g. deliberately added or removed.
        return "–", MISSING_ICON
    diff = new - old
    if diff == 0:
        return "0", ""
    text = f"{int(diff):+d}" if kind == "count" else f"{diff:+.4g}"
    if old == 0:
        return text, CHANGE_ICON
    rel = diff / abs(old)
    flag = CHANGE_ICON if abs(rel) > THRESHOLDS[kind] else ""
    return f"{text} ({rel:+.1%})", flag


def infer_kind(*values):
    present = [value for value in values if value is not None]
    if present and all(
        isinstance(value, int) and not isinstance(value, bool) for value in present
    ):
        return "count"
    return "float"


def baseline_message(status, baseline_url):
    """Explain why no baseline is available, or None if nothing to say."""
    run = (
        f"[latest `main` run]({baseline_url})" if baseline_url else "latest `main` run"
    )
    step = 'See the "Download baseline from main" step log for details.'
    messages = {
        "lookup-failed": f"Could not query previous runs on `main`. {step}",
        "no-run": (
            "No successful push or nightly regression run was found on `main`, "
            "so there is nothing to compare against."
        ),
        "download-failed": (
            f"Could not download the results artifact from the {run} "
            f"(it may have expired). {step}"
        ),
        "missing-summary": f"The {run} did not include a `summary.json`.",
        "unreadable": f"The `summary.json` from the {run} could not be parsed.",
        "unknown": f"The baseline step did not complete. {step}",
    }
    if status not in messages:
        return None
    return f"_No baseline available for comparison: {messages[status]}_"


def render(
    summary, baseline=None, run_url=None, baseline_url=None, baseline_status=None
):
    lines = [MARKER, "## Regression Test Summary", ""]
    if baseline is not None:
        if baseline_url:
            lines += [f"Compared against the [latest `main` run]({baseline_url}).", ""]
    else:
        message = baseline_message(baseline_status, baseline_url)
        if message:
            lines += [message, ""]

    if baseline is None:
        header = ["| Metric | Value |", "|---|---:|"]
    else:
        header = ["| Metric | main | PR | Δ |", "|---|---:|---:|---:|"]
    flags = set()

    def row(label, key, kind):
        new = summary.get(key)
        if baseline is None:
            return f"| {label} | {fmt(new, kind)} |"
        old = baseline.get(key)
        delta, flag = fmt_delta(new, old, kind)
        if flag:
            flags.add(flag)
            delta = f"{delta} {flag}"
        return f"| {label} | {fmt(old, kind)} | {fmt(new, kind)} | {delta} |"

    available = set(summary) | set(baseline or {})
    known = set()
    for title, metrics in SECTIONS.items():
        rows = []
        for key, label, kind in metrics:
            known.add(key)
            if key in available:
                rows.append(row(label, key, kind))
        if rows:
            lines += [f"### {title}", ""] + header + rows + [""]

    extra = sorted(available - known)
    if extra:
        rows = [
            row(
                f"`{key}`", key, infer_kind(summary.get(key), (baseline or {}).get(key))
            )
            for key in extra
        ]
        lines += ["<details><summary>Other metrics</summary>", ""]
        lines += header + rows + ["", "</details>", ""]

    if CHANGE_ICON in flags:
        lines += [
            (
                f"{CHANGE_ICON} marks changes beyond the threshold "
                f"(runtime {THRESHOLDS['time']:.0%}, statistics "
                f"{THRESHOLDS['float']:.0%}, any change in counts). "
                "Runtime on shared runners varies, so treat small timing "
                "changes with caution."
            ),
            "",
        ]
    if MISSING_ICON in flags:
        lines += [
            (
                f"{MISSING_ICON} marks metrics reported by only one of `main` "
                "and this PR. This is expected if the PR adds or removes a "
                "metric; otherwise the metric may have failed to compute."
            ),
            "",
        ]
    if run_url:
        lines += [
            (
                "The profile (`profile.html`) and offset ECDF plot are in the "
                f"`regression-results` artifact of [this run]({run_url})."
            ),
            "",
        ]
    return "\n".join(lines)


def load_baseline(path, status):
    """Return (baseline dict or None, status)."""
    if path is None or status not in (None, "ok"):
        return None, status
    if not path.exists():
        return None, "missing-summary"
    try:
        return json.loads(path.read_text()), "ok"
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, "unreadable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument(
        "--baseline-status",
        default=None,
        help=(
            "Outcome of fetching the baseline: ok, lookup-failed, no-run, "
            "download-failed, unknown, or not-applicable (no comparison wanted)."
        ),
    )
    parser.add_argument("--run-url", default=None)
    parser.add_argument("--baseline-url", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    baseline, status = load_baseline(args.baseline, args.baseline_status)

    report = render(summary, baseline, args.run_url, args.baseline_url, status)
    if args.output is not None:
        args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
