import glob
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pyinstrument
import pytest
from astropy.coordinates import SkyCoord

from sotrplib.config.config import Settings
from sotrplib.sources.sources import MeasuredSource


# a pytest fixture providing "result" that is parameterized over all
# pickle or json files in the current directory
@pytest.fixture(params=glob.glob("./*.pickle"))
def result(request):
    filename = request.param
    with open(filename, "rb") as handle:
        result = pickle.load(handle)
    return result


def test_structure(result):
    assert "forced_photometry" in result
    assert "sifted_blind_search" in result


def test_number_with_photometry(result):
    photometery = result["forced_photometry"]
    assert len(photometery) > 2000


def test_number_found(result):
    """
    Test that our efficiency of finding candidates is high and that
    we don't spuriously find noise or transient candidates.
    """
    blind_results = result["sifted_blind_search"]
    assert len(blind_results.source_candidates) > 2000
    assert len(blind_results.noise_candidates) == 0
    assert len(blind_results.transient_candidates) == 0


def calculate_separation(candidate: MeasuredSource) -> float:
    ra = candidate.ra
    dec = candidate.dec
    ra_offset = candidate.fit_params["ra_offset"]
    dec_offset = candidate.fit_params["dec_offset"]
    this = SkyCoord(ra=ra, dec=dec)
    other = SkyCoord(ra=ra + ra_offset, dec=dec + dec_offset)
    return this.separation(other).deg


def main():
    config = Settings.from_file("regression.json")
    pipeline = config.to_runner()
    with pyinstrument.Profiler() as profiler:
        results = pipeline.run()
    profiler.write_html("profile.html", timeline=True)
    print(profiler.output_text(unicode=True, color=True))
    result = results[0]
    photometry, sifter_result, _ = result

    found = [candidate for candidate in photometry if not candidate.fit_failed]

    dec_offsets = np.array(
        [candidate.fit_params["dec_offset"].value for candidate in found]
    )
    ra_offsets = np.array(
        [candidate.fit_params["ra_offset"].value for candidate in found]
    )
    offsets = np.array([calculate_separation(candidate) for candidate in found])

    plt.ecdf(np.abs(dec_offsets), label="Dec")
    plt.ecdf(np.abs(ra_offsets), label="RA")
    plt.ecdf(np.abs(offsets), label="Total")
    plt.xscale("log")
    plt.xlabel("Offset [deg]")
    plt.ylabel("Cumulative Fraction")
    plt.legend(loc="lower right")
    plt.savefig("offset_ecdf.png")
    plt.close()

    summary = dict(
        duration=profiler.last_session.duration,
        number_with_forced_photometry=len(photometry),
        number_blind_sources=len(sifter_result.source_candidates),
        number_noise_sources=len(sifter_result.noise_candidates),
        number_transient_sources=len(sifter_result.transient_candidates),
        mean_offset_deg=float(np.mean(offsets)),
        median_offset_deg=float(np.median(offsets)),
        std_offset_deg=float(np.std(offsets)),
        mean_ra_offset_deg=float(np.mean(ra_offsets)),
        median_ra_offset_deg=float(np.median(ra_offsets)),
        std_ra_offset_deg=float(np.std(ra_offsets)),
        mean_dec_offset_deg=float(np.mean(dec_offsets)),
        median_dec_offset_deg=float(np.median(dec_offsets)),
        std_dec_offset_deg=float(np.std(dec_offsets)),
    )
    with open("summary.json", "w") as handle:
        json.dump(summary, handle, indent=4)

    import IPython

    IPython.embed()


if __name__ == "__main__":
    main()
