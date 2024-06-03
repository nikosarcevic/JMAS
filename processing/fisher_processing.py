import os
import numpy as np
from numpy.linalg import inv

# Define the base path to the Fisher matrices
base_path = os.path.dirname(os.path.abspath(__file__))
fisher_path = os.path.join(base_path, "../jmas_data/fisher_matrices/")

# List of keys for the Fisher matrices
keys = ["srd_y1", "srd_y10", "jmas_y1", "jmas_y10", "srd+lf_y1", "srd+lf_y10"]

# Load all Fisher matrices
fisher_matrices_all = {key: np.load(os.path.join(fisher_path, f"{key}_cosmic_shear_fisher_matrix.npy")) for key in keys}
fisher_matrices_cosmo = {key: np.load(os.path.join(fisher_path, f"{key}_cosmic_shear_fisher_matrix_cosmo.npy")) for key in keys}


# Create nested dictionaries for Fisher matrices and their inverses
fisher_matrices = {
    "srd": {
        "cosmoia": {"1": fisher_matrices_all["srd_y1"],
                    "10": fisher_matrices_all["srd_y10"]},
        "cosmo": {"1": fisher_matrices_cosmo["srd_y1"],
                  "10": fisher_matrices_cosmo["srd_y10"]}
    },
    "srd+lf": {
        "cosmoialf": {"1": fisher_matrices_all["srd+lf_y1"],
                      "10": fisher_matrices_all["srd+lf_y10"]},
        "cosmo": {"1": fisher_matrices_cosmo["srd+lf_y1"],
                  "10": fisher_matrices_cosmo["srd+lf_y10"]},
        "cosmoia": {"1": fisher_matrices_all["srd+lf_y1"][:11, :11],
                    "10": fisher_matrices_all["srd+lf_y10"][:11, :11]}
    },
    "jmas": {
        "cosmoialf": {"1": fisher_matrices_all["jmas_y1"],
                      "10": fisher_matrices_all["jmas_y10"]},
        "cosmo": {"1": fisher_matrices_cosmo["jmas_y1"],
                  "10": fisher_matrices_cosmo["jmas_y10"]},
        "cosmoia": {"1": fisher_matrices_all["jmas_y1"][:11, :11],
                    "10": fisher_matrices_all["jmas_y10"][:11, :11]}
    }
}


def invert_matrix_structure(matrix_dict):
    return {k: invert_matrix_structure(v) if isinstance(v, dict) else inv(v) for k, v in matrix_dict.items()}


# Invert all Fisher matrices
inverse_fisher_matrices = invert_matrix_structure(fisher_matrices)

inverse_fisher_matrices_clipped = {
    "srd": {
        "cosmoia": {"1": inv(fisher_matrices["srd"]["cosmoia"]["1"]),
                    "10": inv(fisher_matrices["srd"]["cosmoia"]["10"])},
        "cosmo": {"1": inv(fisher_matrices["srd"]["cosmoia"]["1"])[:7, :7],
                  "10": inv(fisher_matrices["srd"]["cosmoia"]["10"])[:7, :7]}
    },
    "srd+lf": {
        "cosmoialf": {"1": inv(fisher_matrices["srd+lf"]["cosmoialf"]["1"]),
                      "10": inv(fisher_matrices["srd+lf"]["cosmoialf"]["10"])},
        "cosmo": {"1": inv(fisher_matrices["srd+lf"]["cosmoialf"]["1"])[:7, :7],
                  "10": inv(fisher_matrices["srd+lf"]["cosmoialf"]["10"])[:7, :7]},
        "cosmoia": {"1": inv(fisher_matrices["srd+lf"]["cosmoialf"]["1"])[:11, :11],
                    "10": inv(fisher_matrices["srd+lf"]["cosmoialf"]["10"])[:11, :11]}
    },
    "jmas": {
        "cosmoialf": {"1": inv(fisher_matrices["jmas"]["cosmoialf"]["1"]),
                      "10": inv(fisher_matrices["jmas"]["cosmoialf"]["10"])},
        "cosmo": {"1": inv(fisher_matrices["jmas"]["cosmoialf"]["1"])[:7, :7],
                  "10": inv(fisher_matrices["jmas"]["cosmoialf"]["10"])[:7, :7]},
        "cosmoia": {"1": inv(fisher_matrices["jmas"]["cosmoialf"]["1"])[:11, :11],
                    "10": inv(fisher_matrices["jmas"]["cosmoialf"]["10"])[:11, :11]}
    }
}

#np.save(f"{fisher_path}fisher_matrices.npy", fisher_matrices)
#np.save(f"{fisher_path}inverse_fisher_matrices.npy", inverse_fisher_matrices)
#np.save(f"{fisher_path}inverse_fisher_matrices_clipped.npy", inverse_fisher_matrices_clipped)
