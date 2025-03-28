import matplotlib.pyplot as plt
import numpy as np

from ..maps.maps import Depth1Map


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


def plot_map_thumbnail(imap,
                  ra:float,
                  dec:float,
                  source_name:str='',
                  thumbnail_width:float = 0.5,
                  plot_dir:str = './',
                  output_file_name:str='',
                  save_thumbnail_map:bool=False,
                  colorbar_range:float=6.0,
                  plot_flux:bool=False,
                 ):
        ## Cut thumbnails from map 
        ## ra,dec in decimal deg
        ## thumbnail_width in deg
        ##
        ## if save_thumbnail_map, then save the .fits map to same directory
        ## with same name (but .fits instead of .png)
        ## colorbar_range is symmetric about 0
        ## plot_flux plots the flux, rather than the default snr
        from pixell import enmap,enplot
        from sotrplib.utils.utils import radec_to_str_name
        from sotrplib.maps.maps import get_thumbnail
        if not source_name:
            source_name = radec_to_str_name(ra,dec).split(' ')[-1] ## just the J000000-000000 part
        
        thumbnail = get_thumbnail(imap,
                                      ra,
                                      dec,
                                      size_deg=thumbnail_width
                                      )
        
        if plot_flux:
            plot_type='flux'
        else:
            plot_type='snr'
        # save maps
        if not output_file_name:
            name = f'{plot_dir}{source_name}_%s_thumbnail'%plot_type
        else:
            name = plot_dir+output_file_name
        if save_thumbnail_map:
            enmap.write_map(name + '.fits', 
                            thumbnail
                            )
            
        # plot
        img = enplot.plot(thumbnail, 
                          range=colorbar_range, 
                          grid=True
                         )
        enplot.write(name, 
                     img
                    )
        
        return


def plot_depth1_thumbnail(imap:Depth1Map,
                          ra:float,
                          dec:float,
                          source_name:str='',
                          thumbnail_width:float = 0.5,
                          plot_dir:str = './',
                          output_file_name:str='',
                          save_thumbnail_map:bool=False,
                          colorbar_range:float=6.0,
                          plot_flux:bool=False,
                         ):
        ## Cut thumbnails from map 
        ## ra,dec in decimal deg
        ## thumbnail_width in deg
        ##
        ## if save_thumbnail_map, then save the .fits map to same directory
        ## with same name (but .fits instead of .png)
        ## colorbar_range is symmetric about 0
        ## plot_flux plots the flux, rather than the default snr
        from pixell import enmap,enplot
        from .utils import radec_to_str_name
        from ...sotrplib.maps.maps import get_thumbnail
        if not source_name:
            source_name = radec_to_str_name(ra,dec).split(' ')[-1] ## just the J000000-000000 part
        
        rho_thumbnail = get_thumbnail(imap.rho_map,
                                      ra_deg,
                                      dec_deg,
                                      size_deg=thumbnail_width
                                      )
        kappa_thumbnail = get_thumbnail(imap.kappa_map,
                                        ra_deg,
                                        dec_deg,
                                        size_deg=thumbnail_width
                                        )
        if plot_flux:
            thumbnail = rho_thumbnail/kappa_thumbnail
            plot_type='flux'
        else:
            thumbnail = rho_thumbnail*kappa_thumbnail**-0.5
            plot_type='snr'
        # save maps
        if not output_file_name:
            name = f'{plot_dir}{source_name}_%s_thumbnail'%plot_type
        else:
            name = plot_dir+output_file_name
        if save_thumbnail_map:
            enmap.write_map(name + '.fits', 
                            thumbnail
                            )
            
        # plot
        img = enplot.plot(thumbnail, 
                          range=colorbar_range, 
                          grid=True
                         )
        enplot.write(name, 
                     img
                    )
        del rho_thumbnail,kappa_thumbnail,thumbnail
        
        return

