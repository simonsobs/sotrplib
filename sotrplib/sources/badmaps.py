import numpy as np

######### rejection function using kappa thresholds


def reject_badmaps(
    kappa_map,
    time,
    ra,
    dec,
    kappa_thresh,
    frac_kappa_thresh10,
    frac_kappa_thresh40,
    frac_low_10x10=0.25,
    frac_low_40x40=0.5,
):
    # sky coordinates to pixel indices
    pos = np.array([dec, ra])  # of agn
    y_f, x_f = kappa_map.sky2pix(pos)  # y,x pixel coordinates (to find in depth1 maps)
    y, x = (
        int(y_f),
        int(x_f),
    )  # for thumbnailing maps later on in this function (array slicing)

    # basically, the 1% of pixels with the lowest inverse variance in the map are treated as candidates for rejection
    # since uncertainty and kappa are inversely proportional, higher kappa -> lower uncertainty whereas lower kappa -> higher uncertainty. So, since we're taking the lowest 1% of kappa here (1st percentile), then only the worst and most unreliable values (that have large uncertainties) are considered the threshold.

    # central pixel (point source)
    k_center = kappa_map.at(pos, mode="nn")[0]

    # thumbnail slices centered on said source
    k10_thumb = kappa_map[y - 5 : y + 5, x - 5 : x + 5]  # 10x10
    k40_thumb = kappa_map[y - 20 : y + 20, x - 20 : x + 20]  # 40x40

    reject = 1  # this means that the anything in the catalog that has 1 in this column is not rejected (by default)
    reject_con = 0
    reject_reason = 0
    statement = ""
    BAD_TIMES = {
        1502427600,
        1502341200,
        1502409600,
    }  # cap was left on during the observations for these times so we don't use them

    if (
        k_center <= kappa_thresh
    ):  # making sure the source is ACTUALLY at the center (equiv. to hits.at(pos) <= 10), rejects map cuz the source location itself is bad quality
        reject = 0
        reject_con = 1
        reject_reason = k_center  # you can probably get rid of this tbh
        statement = "[REJECT due to no center kappa]"

    # 10x10 thumbnail
    # Note that frac_kappa_thresh10 and frac_kappa_thresh40 are defined in the process_info_files function
    elif (
        np.mean(k10_thumb <= frac_kappa_thresh10) >= frac_low_10x10
    ):  # local neighborhood contamination
        # this is basically saying that if you take 25% the pixels in this 10x10 thumbnail and their kappa (inverse variance) values are lower than the lowest 2.5% kappa values in the whole depth-1 map that this thumbnail is from, then it's got even higher noise than that worst 2.5% in the depth-1 map, and should thus be rejected
        # (idk if that made sense but I can draw a little diagram if necessary!)
        reject = 0
        reject_con = 2
        reject_reason = np.mean(k10_thumb <= kappa_thresh)
        statement = "[REJECT due to center 10*10 kappa]"

    # 40x40 thumbnail
    elif (
        np.mean(k40_thumb <= frac_kappa_thresh40) >= frac_low_40x40
    ):  # large scale contamination
        reject = 0
        reject_con = 3
        reject_reason = np.mean(k40_thumb <= kappa_thresh)
        statement = "[REJECT due to center 40*40 kappa]"

    # bad times
    elif int(time) in BAD_TIMES:
        reject = 0
        reject_con = 4
        reject_reason = time
        statement = "[REJECT due to missing source (bad time)]"

    # return after all checks
    return reject, reject_con, reject_reason, statement
