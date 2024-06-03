import numpy as np
from numpy.linalg import inv
import os

# Define the base path to the Fisher matrices
base_path = os.path.dirname(os.path.abspath(__file__))
fisher_path = os.path.join(base_path, "../jmas_data/stability_testing/")

stem_fishers = os.path.join(fisher_path, "stem_derivatives/")
perturbed_fishers = os.path.join(fisher_path, "perturbation_testing/")

# List of keys for the Fisher matrices
keys = ["srd_y1", "srd_y10"]
res_keys = ["res300", "res500", "res1000"]
pert_keys = ["perturbed_res300"]

# Load all Fisher matrices for resolution keys
fisher_matrices_stem_loaded = {
    res_key: {
        key: np.load(os.path.join(stem_fishers, f"{key}_cosmic_shear_fisher_matrix_stem_{res_key}.npy"))
        for key in keys
    }
    for res_key in res_keys
}

# Load all Fisher matrices for perturbed keys
fisher_matrices_pert_loaded = {
    key: {
        pert_key: np.load(os.path.join(perturbed_fishers, f"{key}_cosmic_shear_fisher_matrix_{pert_key}.npy"))
        for pert_key in pert_keys
    }
    for key in keys
}

# Nested dictionary structure for srd
fisher_matrices_stem = {
    "srd": {
        res_key: {
            "cosmoia": {
                "1": fisher_matrices_stem_loaded[res_key]["srd_y1"],
                "10": fisher_matrices_stem_loaded[res_key]["srd_y10"]
            }
        }
        for res_key in res_keys
    }
}

fisher_matrices_pert = {
    "srd": {
        "cosmoia": {
            "1": fisher_matrices_pert_loaded["srd_y1"]["perturbed_res300"],
            "10": fisher_matrices_pert_loaded["srd_y10"]["perturbed_res300"]
        }
    }
}

# Invert all Fisher matrices for stem and perturbed
inverse_fisher_matrices_stem = {
    "srd": {
        res_key: {
            "cosmoia": {
                "1": inv(fisher_matrices_stem["srd"][res_key]["cosmoia"]["1"]),
                "10": inv(fisher_matrices_stem["srd"][res_key]["cosmoia"]["10"])
            }
        }
        for res_key in res_keys
    }
}

inverse_fisher_matrices_pert = {
    "srd": {
        "cosmoia": {
            "1": inv(fisher_matrices_pert["srd"]["cosmoia"]["1"]),
            "10": inv(fisher_matrices_pert["srd"]["cosmoia"]["10"])
        }
    }
}
