from getdist.gaussian_mixtures import GaussianND
import numpy as np
import os
from processing import fisher_stability_processing as fsp

# Define the base path to the corner plot data
base_path = os.path.dirname(os.path.abspath(__file__))
corner_path = os.path.join(base_path, "../jmas_data/corner_plot_data/")

# Load parameters
parameters = np.load(os.path.join(corner_path, "corner_plot_parameters.npy"), allow_pickle=True).item()
inverse_fisher_matrices_stem = fsp.inverse_fisher_matrices_stem
inverse_fisher_matrices_pert = fsp.inverse_fisher_matrices_pert


def create_gaussian_dict_stem(param_set, inv_fisher_set, dataset, res_keys, years):
    """
    Creates a dictionary of gaussians (covariances) that are used for `get_dist` corner plots.
    Stem refers to the fact the derivatives method used in the construction of the Fisher matrix.
    
    Parameters:
    param_set (dict): A dictionary containing parameter values, labels, and names.
    inv_fisher_set (dict): A dictionary containing inverse Fisher matrices.
    dataset (str): The dataset to use.
    res_keys (list): A list of resolution keys (resolution of redshift range z
                    over which the redshift distribution is evaluated). 
                    The values of the z resolution is part of the file name.
    years (list): A list of years.

    Returns
        gaussians (dict): A dictionary contaning the GaussiansND objects from `get_dist`.
    """

    gaussians = {
        res_key: {
            "cosmoia": {
                year: GaussianND(param_set["values"][dataset]["cosmoia"][year],
                                 inv_fisher_set["srd"][res_key]["cosmoia"][year],
                                 labels=param_set["labels"][dataset]["cosmoia"][year],
                                 names=param_set["names"][dataset]["cosmoia"][year])
                for year in years
            }
        }
        for res_key in res_keys
    }
    return gaussians


def create_gaussian_dict_pert(param_set, inv_fisher_set, dataset, years):

    gaussians = {
        "cosmoia": {
            year: GaussianND(param_set["values"][dataset]["cosmoia"][year],
                             inv_fisher_set["srd"]["cosmoia"][year],
                             labels=param_set["labels"][dataset]["cosmoia"][year],
                             names=param_set["names"][dataset]["cosmoia"][year])
            for year in years
        }
    }

    return gaussians


years = ["1", "10"]
res_keys = ["res300", "res500", "res1000"]
srd_keys = ["cosmoia"]

gaussians_stem = {
    "srd": create_gaussian_dict_stem(parameters, inverse_fisher_matrices_stem, "srd", res_keys, years)
}

gaussians_pert = {
    "srd": create_gaussian_dict_pert(parameters, inverse_fisher_matrices_pert, "srd", years)
}
