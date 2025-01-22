import numpy as np
import os.path as op

from pixell import enmap

from astropy.table import Table

def get_sourceflux_threshold(freq):
    """
    Get sourceflux threshold measured from mapnoise * 5 / 5

    avg dflux of f090 = 28.517394966150707
    avg dflux of f150 = 53.679349692751394
    avg dflux of f220 = 88.04350067775873
    avg dflux of f040 = 93.94425336313529
    avg dflux of f030 = 133.4519154035587


    Args:
        freq: frequency

    Returns:
        sourceflux threshold [mJy]
    """
    if freq == "f090":
        sourceflux_thresh = 30.0
    elif freq == "f150":
        sourceflux_thresh = 50.0
    elif freq == "f220":
        sourceflux_thresh = 90.0
    elif freq == "f040":
        sourceflux_thresh = 90.0
    elif freq == "f030":
        sourceflux_thresh = 130.0
    else:
        raise ValueError("Frequency not recognized")
    return sourceflux_thresh


def get_sourcecats():
    sourcecat_f090 = op.join(
        op.dirname(__file__), "../data/inputs/PS_S19_f090_2pass_optimalCatalog.fits"
    )
    sourcecat_f150 = op.join(
        op.dirname(__file__), "../data/inputs/PS_S19_f150_2pass_optimalCatalog.fits"
    )
    sourcecat_f220 = op.join(
        op.dirname(__file__), "../data/inputs/PS_S19_f220_2pass_optimalCatalog.fits"
    )
    sourcecat_f030 = op.join(
        op.dirname(__file__),
        "../data/inputs/cmb_night_pa7_f030_4pass_8way_coadd_map_cat.fits",
    )
    sourcecat_f040 = op.join(
        op.dirname(__file__),
        "../data/inputs/cmb_night_pa7_f040_4pass_8way_coadd_map_cat.fits",
    )

    s040 = Table.read(sourcecat_f040)
    s040["RADeg"] = s040["ra"]
    s040["decDeg"] = s040["dec"]
    # fconv_40 = pixell_utils.dplanck(40.e9, pixell_utils.T_cmb)
    # s040['fluxJy'] = s040['flux'][:, 0] * fconv_40 * 2804.
    s040["fluxJy"] = s040["flux"][:, 0] / 1e3

    s030 = Table.read(sourcecat_f030)
    s030["RADeg"] = s030["ra"]
    s030["decDeg"] = s030["dec"]
    # fconv_30 = pixell_utils.dplanck(30.e9, pixell_utils.T_cmb)
    # s030['fluxJy'] = s030['flux'][0, :] * fconv_30 * 5060.
    s030["fluxJy"] = s030["flux"][:, 0] / 1e3

    sourcecats = {
        "f090": Table.read(sourcecat_f090),
        "f150": Table.read(sourcecat_f150),
        "f220": Table.read(sourcecat_f220),
        "f030": s030,
        "f040": s040,
    }
    return sourcecats

@np.vectorize
def planck_fluxcal(arr, freq):

    """
    Get flux calibration factor for a given frequency

    Args:
        freq: frequency

    Returns:
        flux calibration factor
    """

    # with open('../../data/inputs/calibs_dict_ACTxACT-dr4.pkl', 'rb') as f:
    # dr6_calib = pk.load(f)
    #
    # calib_dict = {'pa4': {'f150': dr6_calib['dr6_pa4_f150']['calibs'], 'f220': dr6_calib['dr6_pa4_f220']['calibs']},
    #       'pa5': {'f090': dr6_calib['dr6_pa5_f090']['calibs'], 'f150': dr6_calib['dr6_pa5_f150']['calibs']},
    #       'pa6': {'f090': dr6_calib['dr6_pa6_f090']['calibs'], 'f150': dr6_calib['dr6_pa6_f150']['calibs']}
    #                 }

    calib_dict = {
        "pa4": {
            "f150": [1.0072175262677963, 0.0023154207035651023],
            "f220": [1.0340185610989223, 0.01626333139583416],
        },
        "pa5": {
            "f090": [1.0218836321586204, 0.0011691303344386838],
            "f150": [0.9876564792184743, 0.001704546067935355],
        },
        "pa6": {
            "f090": [1.0181628559678786, 0.0013096342890986763],
            "f150": [0.9699498005405339, 0.002141045797557799],
        },
        "pa7": {"f030": [1.0, 1.0], "f040": [1.0, 1.0]},
    }  # Alternatively hardcoded

    return calib_dict[arr][freq][0], calib_dict[arr][freq][1]

def get_mean_flux(ra_deg, dec_deg, freq, size):
    """calculates mean flux given position and band

    Args:
        ra_deg: ra of candidate [deg]
        dec_deg:dec of candidates [deg]
        freq: frequency band, i.e. f220,f150 etc
    Returns:
        flux [mJy], snr
    """
    from filters import matched_filter_1overf
    ra = ra_deg * np.pi / 180.0
    dec = dec_deg * np.pi / 180.0
    mean_map_file = (
        "/home/snaess/project/actpol/map_coadd/20211219/release/act_daynight_%s_map.fits"
        % freq
    )
    mean_ivar_file = (
        "/home/snaess/project/actpol/map_coadd/20211219/release/act_daynight_%s_ivar.fits"
        % freq
    )
    mean_map = enmap.read_map(
        mean_map_file, box=[[dec - size, ra - size], [dec + size, ra + size]]
    )[0, :, :]
    mean_ivar = enmap.read_map(
        mean_ivar_file, box=[[dec - size, ra - size], [dec + size, ra + size]]
    )
    rho, kappa = matched_filter_1overf(mean_map, mean_ivar, freq, size_deg=0.5)
    flux_map = rho / kappa
    snr_map = rho / kappa**0.5
    flux_map[np.where(kappa == 0.0)] = 0.0
    snr_map[np.where(kappa == 0.0)] = 0.0
    flux = flux_map.at([dec, ra])
    snr = snr_map.at([dec, ra])
    return flux, snr

def merge_result(flux_data):  # merge pa4 and pa5 result
    result_merged = np.array(
        [[225.0 * 1e9, 0.0, 0.0], [150.0 * 1e9, 0.0, 0.0], [98.0 * 1e9, 0.0, 0.0]]
    )
    result_merged[0, 1] = flux_data["pa4_f220_flux"][0]
    result_merged[0, 2] = flux_data["pa4_f220_flux"][1]
    result_merged[2, 1] = flux_data["pa5_f090_flux"][0]
    result_merged[2, 2] = flux_data["pa5_f090_flux"][1]
    ivar_150 = 0.0
    for key in ["pa4_f150_flux", "pa5_f150_flux"]:
        if flux_data[key][1] != 0:
            result_merged[1, 1] += flux_data[key][0] / flux_data[key][1] ** 2.0
            ivar_150 += 1 / flux_data[key][1] ** 2.0
    if ivar_150 != 0.0:
        result_merged[1, 1] /= ivar_150
        result_merged[1, 2] = 1 / ivar_150**0.5
    return result_merged

def merge_result_all(flux_data):  # merge same band results from all array
    result_merged = np.array(
        [
            [225.0 * 1e9, 0.0, 0.0],
            [150.0 * 1e9, 0.0, 0.0],
            [98.0 * 1e9, 0.0, 0.0],
            [40 * 1e9, 0.0, 0.0],
            [30 * 1e9, 0.0, 0.0],
        ]
    )
    result_merged[0, 1] = flux_data["pa4_f220_flux"][0]
    result_merged[0, 2] = flux_data["pa4_f220_flux"][1]
    result_merged[3, 1] = flux_data["pa7_f040_flux"][0]
    result_merged[3, 2] = flux_data["pa7_f040_flux"][1]
    result_merged[4, 1] = flux_data["pa7_f030_flux"][0]
    result_merged[4, 2] = flux_data["pa7_f030_flux"][1]
    ivar_150 = 0.0
    for key in ["pa4_f150_flux", "pa5_f150_flux", "pa6_f150_flux"]:
        if flux_data[key][1] != 0:
            result_merged[1, 1] += flux_data[key][0] / flux_data[key][1] ** 2.0
            ivar_150 += 1 / flux_data[key][1] ** 2.0
    if ivar_150 != 0.0:
        result_merged[1, 1] /= ivar_150
        result_merged[1, 2] = 1 / ivar_150**0.5
    ivar_090 = 0.0
    for key in ["pa5_f090_flux", "pa6_f090_flux"]:
        if flux_data[key][1] != 0:
            result_merged[2, 1] += flux_data[key][0] / flux_data[key][1] ** 2.0
            ivar_090 += 1 / flux_data[key][1] ** 2.0
    if ivar_090 != 0.0:
        result_merged[2, 1] /= ivar_090
        result_merged[2, 2] = 1 / ivar_090**0.5
    return result_merged

def get_spectra_index(flux_data, ctime, data_type="pa4pa5_or_third"):
    def func(x, a, b):
        y = a * x**b
        return y
    from .utils import calculate_alpha
    from scipy.interpolate import curve_fit
    if data_type == "pa4pa5_or_third":
        merged_data = merge_result(flux_data)
        result_merged_sel = merged_data[np.where(merged_data[:, 1] > 0.0)]
        if result_merged_sel.shape[0] == 3:
            pars, cov = curve_fit(
                f=func,
                xdata=result_merged_sel[:, 0],
                ydata=result_merged_sel[:, 1],
                sigma=result_merged_sel[:, 2],
                absolute_sigma=True,
                p0=[0, 1],
                maxfev=5000,
            )
            alpha = pars[1]
            err = np.sqrt(cov[1, 1])
        elif result_merged_sel.shape[0] == 2:
            amp, alpha, err = calculate_alpha(result_merged_sel)
        else:
            if ctime < 1580515200:
                pa6_data = np.array(
                    [
                        [
                            150 * 1e9,
                            flux_data["pa6_f150_flux"][0],
                            flux_data["pa6_f150_flux"][1],
                        ],
                        [
                            98 * 1e9,
                            flux_data["pa6_f090_flux"][0],
                            flux_data["pa6_f090_flux"][1],
                        ],
                    ]
                )
                pa6_data_sel = pa6_data[np.where(pa6_data[:, 1] > 0.0)]
                if pa6_data_sel.shape[0] == 2:
                    amp, alpha, err = calculate_alpha(pa6_data_sel)
                else:
                    alpha = 0
                    err = 0
            else:
                pa7_data = np.array(
                    [
                        [
                            40 * 1e9,
                            flux_data["pa7_f040_flux"][0],
                            flux_data["pa7_f040_flux"][1],
                        ],
                        [
                            30 * 1e9,
                            flux_data["pa7_f030_flux"][0],
                            flux_data["pa7_f030_flux"][1],
                        ],
                    ]
                )
                pa7_data_sel = pa7_data[np.where(pa7_data[:, 1] > 0.0)]
                if pa7_data_sel.shape[0] == 2:
                    amp, alpha, err = calculate_alpha(pa7_data_sel)
                else:
                    alpha = 0
                    err = 0
    if data_type == "all":
        merged_data = merge_result_all(flux_data)
        result_merged_sel = merged_data[np.where(merged_data[:, 1] > 0.0)]
        if result_merged_sel.shape[0] > 2:
            pars, cov = curve_fit(
                f=func,
                xdata=result_merged_sel[:, 0],
                ydata=result_merged_sel[:, 1],
                sigma=result_merged_sel[:, 2],
                absolute_sigma=True,
                p0=[0, 1],
                maxfev=5000,
            )
            alpha = pars[1]
            err = np.sqrt(cov[1, 1])
        elif result_merged_sel.shape[0] == 2:
            _, alpha, err = calculate_alpha(result_merged_sel)
        else:
            alpha = 0
            err = 0
    return alpha, err

