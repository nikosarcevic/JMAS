from getdist.gaussian_mixtures import GaussianND
import numpy as np
import os
from processing import fisher_processing as fp

# Define the base path to the corner plot data
base_path = os.path.dirname(os.path.abspath(__file__))
corner_path = os.path.join(base_path, "../jmas_data/corner_plot_data/")

# Load parameters
parameters = np.load(os.path.join(corner_path, "corner_plot_parameters.npy"), allow_pickle=True).item()
inverse_fisher_matrices = fp.inverse_fisher_matrices
inverse_fisher_matrices_clipped = fp.inverse_fisher_matrices_clipped


def create_gaussian_dict(param_set, inv_fisher_set, dataset, keys):
    return {
        key: {
            "1": GaussianND(param_set["values"][dataset][key]["1"],
                            inv_fisher_set[dataset][key]["1"],
                            labels=param_set["labels"][dataset][key]["1"],
                            names=param_set["names"][dataset][key]["1"]),
            "10": GaussianND(param_set["values"][dataset][key]["10"],
                             inv_fisher_set[dataset][key]["10"],
                             labels=param_set["labels"][dataset][key]["10"],
                             names=param_set["names"][dataset][key]["10"])
        } for key in keys
    }

# Define the keys for each dataset
srd_keys = ["cosmoia", "cosmo"]
common_keys = ["cosmoia", "cosmo", "cosmoialf"]

gaussians = {
    "srd": create_gaussian_dict(parameters, inverse_fisher_matrices, "srd", srd_keys),
    "jmas": create_gaussian_dict(parameters, inverse_fisher_matrices, "jmas", common_keys),
    "srd+lf": create_gaussian_dict(parameters, inverse_fisher_matrices, "srd+lf", common_keys)
}

gaussians_clipped = {
    "srd": create_gaussian_dict(parameters, inverse_fisher_matrices_clipped, "srd", srd_keys),
    "jmas": create_gaussian_dict(parameters, inverse_fisher_matrices_clipped, "jmas", common_keys),
    "srd+lf": create_gaussian_dict(parameters, inverse_fisher_matrices_clipped, "srd+lf", common_keys)
}

# Save the Gaussian objects
#path = "jmas_data/corner_plot_data/"
#np.save(f"{path}gaussians.npy", gaussians)
#np.save(f"{path}gaussians_clipped.npy", gaussians_clipped)