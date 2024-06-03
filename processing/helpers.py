from getdist import plots
import numpy as np


def get_stats(cov_matrix, fiducial_values):
    """
    Calculate the means and standard deviations from an inverse covariance matrix and fiducial values.

    Parameters:
    cov_matrix (numpy.ndarray): Covariance matrix (inverse of the Fisher matrix).
    fiducial_values (list or numpy.ndarray): List or array of fiducial values.

    Returns:
    tuple: Tuple containing two numpy arrays:
           - means (numpy.ndarray): Calculated mean values.
           - stds (numpy.ndarray): Calculated standard deviations.
    """
    # Calculate standard deviations from the diagonal of the inverse matrix
    stds = np.sqrt(np.diag(cov_matrix))
    # Means are simply the fiducial values provided
    means = fiducial_values

    return np.array(means), np.array(stds)


def get_fractional_difference(sigma, sigma_benchmark):
    """
    Calculate the percentage difference of two sets of standard deviations.

    Parameters:
    sigma (numpy.ndarray): Array of standard deviations.
    sigma_benchmark (numpy.ndarray): Array of benchmark standard deviations.

    Returns:
    numpy.ndarray: Array of percentage differences.
    """
    # Calculate the absolute percentage difference
    perc_diffs = np.abs((sigma - sigma_benchmark) / sigma_benchmark) * 100
    return np.array(perc_diffs)


def get_relative_difference(sigma, sigma_benchmark):
    """
    Calculate the relative difference between two sets of standard deviations.

    Parameters:
    sigma (numpy.ndarray): Array of standard deviations.
    sigma_benchmark (numpy.ndarray): Array of benchmark standard deviations.

    Returns:
    numpy.ndarray: Array of relative differences.
    """
    # Calculate the relative difference (expressed as a percentage)
    rel_diff = 1 - sigma / sigma_benchmark * 100
    return np.array(rel_diff)


def get_symmetric_percentage_difference(sigma, sigma_benchmark):
    """
    Calculate the symmetric percentage difference between two sets of standard deviations.

    Parameters:
    sigma (numpy.ndarray): Array of standard deviations.
    sigma_benchmark (numpy.ndarray): Array of benchmark standard deviations.

    Returns:
    numpy.ndarray: Array of symmetric percentage differences.
    """
    # Calculate the symmetric percentage difference
    symm_diff = 2 * (sigma - sigma_benchmark) / (np.abs(sigma) + np.abs(sigma_benchmark)) * 100
    return np.array(symm_diff)


def get_peak_and_zpeak(nz, redshift):
    """
    Computes the peak value and zpeak (redshift at the peak of the distribution)
    for a given distribution 'nz'.

    Parameters:
    nz (array): The redshift distribution.
    redshift (array): The redshift values.

    Returns:
        float: The redshift value at the peak of the distribution.
    """
    max_idx = np.argmax(nz)
    return redshift[max_idx], nz[max_idx]


jmas_colors = {
    "srd_y1": "#3d5902",
    "srd_y10": "#134b5f",
    "jmas_y1": "#79b203",
    "jmas_y10": "#2596be",
}

corner_colors = {
    "srd+lf": {
        "1": "#CCFF00",
        "10": "#00FFFF",
    },
    "srd": {
        "1": "#3d5902",
        "10": "#134b5f",
    },
    "jmas": {
        "1": "#79b203",
        "10": "#2596be",
    },
}

greens = ["#9cc44d", "#87b726", "#72aa00", "#619100", "#507700"]
corals = ["#f76969", "#f64848", "#f42828", "#cf2222", "#ab1c1c"]
teals = ["#4d98a5", "#268292", "#006c7f", "#005c6c", "#004c59"]
oranges = ["#ffa557", "#ff9233", "#ff7f0f", "#d96c0d", "#b3590b"]
purples = ["#ad8097", "#9c6481", "#8a496b", "#753e5b", "#61334b"]
mycolors = greens + corals + teals + oranges + purples

cosmoplot = plots.get_subplot_plotter(width_inch=10)
cosmoplot.settings.alpha_filled_add = 0.8
cosmoplot.settings.axes_labelsize = 50
cosmoplot.settings.legend_rect_border = False
cosmoplot.settings.axes_fontsize = 35
cosmoplot.settings.figure_legend_frame = False
cosmoplot.settings.legend_fontsize = 50
cosmoplot.settings.linewidth_contour = 3.5
cosmoplot.settings.linewidth = 3.5
cosmoplot.settings.axis_marker_lw = 2
cosmoplot.settings.legend_frac_subplot_margin = 0.1

cosmoiaplot = plots.get_subplot_plotter(width_inch=12)
cosmoiaplot.settings.alpha_filled_add = 0.8
cosmoiaplot.settings.axes_labelsize = 45
cosmoiaplot.settings.legend_rect_border = False
cosmoiaplot.settings.axes_fontsize = 30
cosmoiaplot.settings.figure_legend_frame = False
cosmoiaplot.settings.legend_fontsize = 45
cosmoiaplot.settings.linewidth_contour = 4
cosmoiaplot.settings.linewidth = 4
cosmoiaplot.settings.axis_marker_lw = 2
cosmoiaplot.settings.legend_frac_subplot_margin = 0.1

gdplot = plots.get_subplot_plotter(width_inch=12)
gdplot.settings.alpha_filled_add = 0.7
gdplot.settings.axes_labelsize = 20
gdplot.settings.legend_rect_border = False
gdplot.settings.axes_fontsize = 15
gdplot.settings.figure_legend_frame = False
gdplot.settings.legend_fontsize = 20
gdplot.settings.linewidth_contour = 1.5
gdplot.settings.linewidth = 1.5
gdplot.settings.axis_marker_lw = 0.7
gdplot.settings.legend_frac_subplot_margin = 0.1

dpi = {
    "pdf": 150,
    "png": 300,
}
