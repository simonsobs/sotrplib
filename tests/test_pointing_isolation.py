from astropy import units as u

from sotrplib.config.forced_photometry import Lmfit2DGaussianPointingConfig
from sotrplib.sources.force import TwoDGaussianPointingFitter
from sotrplib.sources.sources import MeasuredSource, RegisteredSource


def _src(ra, dec, flux_jy, name):
    return RegisteredSource(
        ra=ra * u.deg, dec=dec * u.deg, flux=flux_jy * u.Jy, source_id=name
    )


def test_isolation_cut_uses_faint_neighbours_and_radius():
    fitter = TwoDGaussianPointingFitter(
        min_flux=0.3 * u.Jy,
        near_source_rel_flux_limit=0.2,
        near_source_radius=20 * u.arcmin,
    )
    isolated = _src(10.0, 0.0, 1.0, "isolated")
    crowded = _src(20.0, 0.0, 1.0, "crowded")
    # 0.25 Jy is below min_flux but above 0.2 * 1 Jy, 15' away: crowded is cut.
    near_faint = _src(20.25, 0.0, 0.25, "near_faint")
    # 0.15 Jy at 5' from "isolated" is below 0.2 * 1 Jy: no effect.
    near_weak = _src(10.0, 5 / 60, 0.15, "near_weak")
    # A bright source 30' away is outside the radius.
    far = _src(10.5, 0.0, 5.0, "far")
    pointing = [isolated, crowded, far]
    everything = pointing + [near_faint, near_weak]
    flagged = fitter._has_nearby_sources(pointing, everything)
    assert flagged.tolist() == [False, True, False]


def test_isolation_cut_default_radius_is_thumbnail_diagonal():
    fitter = TwoDGaussianPointingFitter(
        thumbnail_half_width=0.1 * u.deg, near_source_rel_flux_limit=0.3
    )
    a = _src(10.0, 0.0, 1.0, "a")
    b = _src(10.0, 10 / 60, 0.5, "b")  # 10' > 8.5'
    assert fitter._has_nearby_sources([a], [a, b]).tolist() == [False]
    b = _src(10.0, 8 / 60, 0.5, "b")
    assert fitter._has_nearby_sources([a], [a, b]).tolist() == [True]


def test_pointing_config_defaults_to_act_isolation_cut():
    fitter = Lmfit2DGaussianPointingConfig().to_forced_photometry()
    assert fitter.near_source_rel_flux_limit == 0.2
    assert fitter.near_source_radius == 20 * u.arcmin


def _fit(fwhm_ra, fwhm_dec):
    return MeasuredSource(
        ra=10 * u.deg,
        dec=0 * u.deg,
        flux=1 * u.Jy,
        fwhm_ra=fwhm_ra * u.arcmin if fwhm_ra is not None else None,
        fwhm_dec=fwhm_dec * u.arcmin if fwhm_dec is not None else None,
    )


def test_point_source_cut_removes_and_marks_extended_fits():
    fitter = TwoDGaussianPointingFitter(max_fwhm_ratio=1.5)
    point = _fit(1.2, 1.3)
    wide = _fit(1.2, 2.0)  # 2.0' > 1.5 * 1.19'
    unknown = _fit(None, None)
    kept = fitter._remove_extended([point, wide, unknown], 1.19 * u.arcmin)
    assert kept == [point, unknown]
    assert wide.extended is True


def test_point_source_cut_is_off_by_default():
    fitter = TwoDGaussianPointingFitter()
    wide = _fit(5.0, 5.0)
    assert fitter._remove_extended([wide], 1.0 * u.arcmin) == [wide]


def test_pointing_config_defaults_to_point_source_cut():
    fitter = Lmfit2DGaussianPointingConfig().to_forced_photometry()
    assert fitter.max_fwhm_ratio == 1.5
    assert fitter.reference_fwhm["f220"] == 1.19 * u.arcmin
