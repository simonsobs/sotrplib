import numpy as np
import matplotlib.pyplot as plt


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
                f"text {dec[i]} {ra[i]} {size[i]/2} {size[i]/2} {label[i]} 60 {label_color} \n"
            )

    # write to file
    if fname is not None:
        with open(fname, "w") as f:
            for line in lines:
                f.writelines(line)

    return lines
