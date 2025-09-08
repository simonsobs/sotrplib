import matplotlib.pyplot as plt
import numpy as np


def fancy_plot(fontsize=25):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams.update({"font.size": fontsize})
    return


def hist(data, binsize, xlabel=None, max=None):
    if max is None:
        max = np.max(data)

    # bins
    bins = np.arange(0, max + binsize, binsize)

    # All data
    fig, ax = plt.subplots()
    hist, bin_edges = np.histogram(data, bins, range=(0, max))
    bincenters = [
        (bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in range(len(bin_edges) - 1)
    ]
    # replace first bincenter with zero
    bincenters[0] = 0
    plt.step(
        bincenters,
        hist / len(data),
        where="mid",
        color="black",
        label=f"All sources ({len(data)})",
    )

    ax.set_ylabel("fraction of data")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.legend()

    return fig, ax


def enplot_annotate(
    ra, dec, size=5, width=5, color="black", label=None, label_color="black", fname=None
):
    """

    Write file to use as input for enplot to annotate a map with circles and text

    Args:
        fname: str, name of file to write to
        ra: list of ra values [deg]
        dec: list of dec values [deg]
        size: list of sizes for circles [pix]
        label: list of labels for circles

    Returns:
        Writes file to fname
    """

    # parse sizes: if scalar, convert to list
    if np.isscalar(size):
        size = [size for i in range(len(ra))]
    if np.isscalar(width):
        width = [width for i in range(len(ra))]

    lines = []
    for i in range(len(ra)):
        # write to file
        lines.append(f"circle {dec[i]} {ra[i]} 0 0 {size[i]} {width[i]} {color} \n")
        if label is not None:
            lines.append(
                f"text {dec[i]} {ra[i]} {size[i] / 2} {size[i] / 2} {label[i]} 60 {label_color} \n"
            )

    # write to file
    if fname is not None:
        with open(fname, "w") as f:
            for line in lines:
                f.writelines(line)

    return lines


def plot_map_thumbnail(
    imap,
    ra: float,
    dec: float,
    source_name: str = "",
    thumbnail_width: float = 0.5,
    plot_dir: str = "./",
    output_file_name: str = "",
    save_thumbnail_map: bool = False,
    colorbar_range: float = 6.0,
    plot_flux: bool = False,
):
    ## Cut thumbnails from map
    ## ra,dec in decimal deg
    ## thumbnail_width in deg
    ##
    ## if save_thumbnail_map, then save the .fits map to same directory
    ## with same name (but .fits instead of .png)
    ## colorbar_range is symmetric about 0
    ## plot_flux plots the flux, rather than the default snr
    from pixell import enmap, enplot

    from sotrplib.maps.maps import get_thumbnail
    from sotrplib.utils.utils import radec_to_str_name

    if not source_name:
        source_name = radec_to_str_name(ra, dec).split(" ")[
            -1
        ]  ## just the J000000-000000 part

    thumbnail = get_thumbnail(imap, ra, dec, size_deg=thumbnail_width)

    if plot_flux:
        plot_type = "flux"
    else:
        plot_type = "snr"
    # save maps
    if not output_file_name:
        name = f"{plot_dir}{source_name}_%s_thumbnail" % plot_type
    else:
        name = plot_dir + output_file_name
    if save_thumbnail_map:
        enmap.write_map(name + ".fits", thumbnail)

    # plot
    img = enplot.plot(thumbnail, range=colorbar_range, grid=True)
    enplot.write(name, img)

    return


def plot_depth1_thumbnail(
    imap,
    ra: float,
    dec: float,
    source_name: str = "",
    thumbnail_width: float = 0.5,
    plot_dir: str = "./",
    output_file_name: str = "",
    save_thumbnail_map: bool = False,
    colorbar_range: float = 6.0,
    plot_flux: bool = False,
):
    ## Cut thumbnails from map
    ## ra,dec in decimal deg
    ## thumbnail_width in deg
    ##
    ## if save_thumbnail_map, then save the .fits map to same directory
    ## with same name (but .fits instead of .png)
    ## colorbar_range is symmetric about 0
    ## plot_flux plots the flux, rather than the default snr
    from pixell import enmap, enplot

    from ...sotrplib.maps.maps import get_thumbnail
    from .utils import radec_to_str_name

    if not source_name:
        source_name = radec_to_str_name(ra, dec).split(" ")[
            -1
        ]  ## just the J000000-000000 part

    # NOTE THAT THESE MIGHT BE WRONG
    ra_deg = ra
    dec_deg = dec

    rho_thumbnail = get_thumbnail(
        imap.rho_map, ra_deg, dec_deg, size_deg=thumbnail_width
    )
    kappa_thumbnail = get_thumbnail(
        imap.kappa_map, ra_deg, dec_deg, size_deg=thumbnail_width
    )
    if plot_flux:
        thumbnail = rho_thumbnail / kappa_thumbnail
        plot_type = "flux"
    else:
        thumbnail = rho_thumbnail * kappa_thumbnail**-0.5
        plot_type = "snr"
    # save maps
    if not output_file_name:
        name = f"{plot_dir}{source_name}_%s_thumbnail" % plot_type
    else:
        name = plot_dir + output_file_name
    if save_thumbnail_map:
        enmap.write_map(name + ".fits", thumbnail)

    # plot
    img = enplot.plot(thumbnail, range=colorbar_range, grid=True)
    enplot.write(name, img)
    del rho_thumbnail, kappa_thumbnail, thumbnail

    return


def ascii_scatter(x_vals, y_vals, width=40, height=20):
    if len(x_vals) != len(y_vals):
        raise ValueError("x_vals and y_vals must be non-empty and of equal length.")

    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)

    def scale(val, min_val, max_val, size):
        if max_val == min_val:
            return 0
        return int((val - min_val) / (max_val - min_val) * (size - 1))

    # Determine axis positions
    origin_xi = scale(0, min_x, max_x, width) if min_x <= 0 <= max_x else None
    origin_yi = (
        height - 1 - scale(0, min_y, max_y, height) if min_y <= 0 <= max_y else None
    )

    # Create canvas
    canvas = [[" " for _ in range(width)] for _ in range(height)]

    # Draw points
    for x, y in zip(x_vals, y_vals):
        xi = scale(x, min_x, max_x, width)
        yi = height - 1 - scale(y, min_y, max_y, height)
        canvas[yi][xi] = "*"

    # Draw axes
    for y in range(height):
        if origin_xi is not None:
            if canvas[y][origin_xi] == "*":
                pass
            else:
                canvas[y][origin_xi] = "|"
    if origin_yi is not None:
        for x in range(width):
            if canvas[origin_yi][x] == "*":
                pass
            elif canvas[origin_yi][x] == "|":
                pass
            else:
                canvas[origin_yi][x] = "-"

    # Print canvas with Y-axis labels
    for i, row in enumerate(canvas):
        y_val = max_y - i * (max_y - min_y) / (height - 1)
        if i % 2:
            print(f"{y_val:>7.2f} | {''.join(row)}")
        else:
            print(f"{' ' * 7} | {''.join(row)}")
    # X-axis line and labels
    print(" " * 8 + "-" * width)
    label_positions = [0, width // 2, width - 1]
    label_values = [min_x, (min_x + max_x) / 2, max_x]
    label_line = [" "] * (width + 9)
    for pos in label_positions:
        label_line[9 + pos] = "|"
    print("".join(label_line))

    # Print X-axis numeric labels
    label_str = f"{' ' * 9}{min_x:.2f}".ljust(9 + width // 2)
    label_str += f"{(min_x + max_x) / 2:.2f}".center(width // 10)
    label_str = label_str[: 9 + width - len(f"{max_x:.2f}")].ljust(
        9 + width - len(f"{max_x:.2f}")
    )
    label_str += f"{max_x:.2f}"
    print(label_str)


def ascii_vertical_histogram(data, bin_width=0.1, min_val=-1.0, max_val=1.0, height=10):
    # Define bins
    num_bins = int((max_val - min_val) / bin_width)
    bins = [0] * num_bins

    # Bin the data
    for val in data:
        if min_val <= val < max_val:
            idx = int((val - min_val) / bin_width)
            bins[idx] += 1
        elif val == max_val:
            bins[-1] += 1

    max_count = max(bins) if bins else 1  # Avoid division by zero

    # Build plot from top down
    for level in reversed(range(1, height + 1)):
        threshold = max_count * level / height
        row = ""
        for count in bins:
            row += " # " if count >= threshold else "   "
        print(f"{row}")

    # Print x-axis
    print("---" * num_bins)

    # Print bin centers
    bin_labels = [f"{min_val + (i + 0.5) * bin_width:.0f}" for i in range(num_bins)]
    label_row = ""
    for lbl in bin_labels:
        label_row += lbl.center(3)
    print(label_row)
