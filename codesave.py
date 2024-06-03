
from getdist.gaussian_mixtures import GaussianND
from getdist import plots
import numpy as np
from numpy.linalg import inv
import pandas as pd
import yaml
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = False
import cmasher as cmr
import pyccl as ccl
import scipy as sp
import getdist as gd
import h5py
import helpers as h


parameters = np.load("jmas_data/fisher_matrices/corner_plot_parameters.npy",
                     allow_pickle=True).item()

import numpy as np
from numpy.linalg import inv

# Define the path to the Fisher matrices
fisher_path = "jmas_data/fisher_matrices/"

# Load Fisher matrices
# List of keys for the Fisher matrices
keys = ["srd_y1", "srd_y10", "jmas_y1", "jmas_y10", "srd+lf_y1", "srd+lf_y10"]

# Load all Fisher matrices
fisher_matrices_all = {key: np.load(f"{fisher_path}{key}_cosmic_shear_fisher_matrix.npy") for key in keys}
fisher_matrices_cosmo = {key: np.load(f"{fisher_path}{key}_cosmic_shear_fisher_matrix_cosmo.npy") for key in keys}

# Calculate the inverse matrices
inverse_fisher_matrices_all = {key: inv(matrix) for key, matrix in fisher_matrices_all.items()}
inverse_fisher_matrices_cosmo = {key: inv(matrix) for key, matrix in fisher_matrices_cosmo.items()}

gaussians = {
    "srd":
        {
            "all":
                {
                    "1": GaussianND(
                        parameters["values"]["srd"]["all"]["1"],
                        inverse_fisher_matrices["srd"]["all"]["1"],
                        labels=parameters["labels"]["srd"]["all"]["1"],
                        names=parameters["names"]["srd"]["all"]["1"]),
                    "10": GaussianND(
                        parameters["values"]["srd"]["all"]["10"],
                        inverse_fisher_matrices["srd"]["all"]["10"],
                        labels=parameters["labels"]["srd"]["all"]["10"],
                        names=parameters["names"]["srd"]["all"]["10"])
                },
            "cosmo":
                {
                    "1": GaussianND(
                        parameters["values"]["srd"]["cosmo"]["1"],
                        inverse_fisher_matrices["srd"]["cosmo"]["1"],
                        labels=parameters["labels"]["srd"]["cosmo"]["1"],
                        names=parameters["names"]["srd"]["cosmo"]["1"]),
                    "10": GaussianND(
                        parameters["values"]["srd"]["cosmo"]["10"],
                        inverse_fisher_matrices["srd"]["cosmo"]["1"],
                        labels=parameters["labels"]["srd"]["cosmo"]["10"],
                        names=parameters["names"]["srd"]["cosmo"]["10"], )
                }
        },
    "jmas":
        {
            "all":
                {
                    "1": GaussianND(
                        parameters["values"]["jmas"]["all"]["1"],
                        inverse_fisher_matrices["jmas"]["all"]["1"],
                        labels=parameters["labels"]["jmas"]["all"]["1"],
                        names=parameters["names"]["jmas"]["all"]["1"], ),
                    "10": GaussianND(
                        parameters["values"]["jmas"]["all"]["10"],
                        inverse_fisher_matrices["jmas"]["all"]["10"],
                        labels=parameters["labels"]["jmas"]["all"]["10"],
                        names=parameters["names"]["jmas"]["all"]["10"], )
                },
            "cosmo":
                {
                    "1": GaussianND(
                        parameters["values"]["jmas"]["cosmo"]["1"],
                        inverse_fisher_matrices["jmas"]["cosmo"]["1"],
                        labels=parameters["labels"]["jmas"]["cosmo"]["1"],
                        names=parameters["names"]["jmas"]["cosmo"]["1"], ),
                    "10": GaussianND(
                        parameters["values"]["jmas"]["cosmo"]["10"],
                        inverse_fisher_matrices["jmas"]["cosmo"]["1"],
                        labels=parameters["labels"]["jmas"]["cosmo"]["10"],
                        names=parameters["names"]["jmas"]["cosmo"]["10"], )
                },
            "cosmoia":
                {
                    "1": GaussianND(
                        parameters["values"]["jmas"]["cosmoia"]["1"],
                        inverse_fisher_matrices["jmas"]["cosmoia"]["1"],
                        labels=parameters["labels"]["jmas"]["cosmoia"]["1"],
                        names=parameters["names"]["jmas"]["cosmoia"]["1"], ),
                    "10": GaussianND(
                        parameters["values"]["jmas"]["cosmoia"]["10"],
                        inverse_fisher_matrices["jmas"]["cosmoia"]["10"],
                        labels=parameters["labels"]["jmas"]["cosmoia"]["10"],
                        names=parameters["names"]["jmas"]["cosmoia"]["10"], )
                },
        },
    "srd+lf":
        {
            "all":
                {
                    "1": GaussianND(
                        parameters["values"]["srd+lf"]["all"]["1"],
                        inverse_fisher_matrices["srd+lf"]["all"]["1"],
                        labels=parameters["labels"]["srd+lf"]["all"]["1"],
                        names=parameters["names"]["srd+lf"]["all"]["1"], ),
                    "10": GaussianND(
                        parameters["values"]["srd+lf"]["all"]["10"],
                        inverse_fisher_matrices["srd+lf"]["all"]["10"],
                        labels=parameters["labels"]["srd+lf"]["all"]["10"],
                        names=parameters["names"]["srd+lf"]["all"]["10"], )
                },
            "cosmo":
                {
                    "1": GaussianND(
                        parameters["values"]["srd+lf"]["cosmo"]["1"],
                        inverse_fisher_matrices["srd+lf"]["cosmo"]["1"],
                        labels=parameters["labels"]["srd+lf"]["cosmo"]["1"],
                        names=parameters["names"]["srd+lf"]["cosmo"]["1"], ),
                    "10": GaussianND(
                        parameters["values"]["srd+lf"]["cosmo"]["10"],
                        inverse_fisher_matrices["srd+lf"]["cosmo"]["1"],
                        labels=parameters["labels"]["srd+lf"]["cosmo"]["10"],
                        names=parameters["names"]["srd+lf"]["cosmo"]["10"], )
                },
            "cosmoia":
                {
                    "1": GaussianND(
                        parameters["values"]["srd+lf"]["cosmoia"]["1"],
                        inverse_fisher_matrices["srd+lf"]["cosmoia"]["1"],
                        labels=parameters["labels"]["srd+lf"]["cosmoia"]["1"],
                        names=parameters["names"]["srd+lf"]["cosmoia"]["1"], ),
                    "10": GaussianND(
                        parameters["values"]["srd+lf"]["cosmoia"]["10"],
                        inverse_fisher_matrices["srd+lf"]["cosmoia"]["10"],
                        labels=parameters["labels"]["srd+lf"]["cosmoia"]["10"],
                        names=parameters["names"]["srd+lf"]["cosmoia"]["10"], )
                }
        }

}

np.save(f"{fisher_path}gaussians.npy", gaussians)




import numpy as np
from numpy.linalg import inv

# Define the path to the Fisher matrices
fisher_path = "jmas_data/fisher_matrices/"

# List of keys for the Fisher matrices
keys = ["srd_y1", "srd_y10", "jmas_y1", "jmas_y10", "srd+lf_y1", "srd+lf_y10"]

# Load all Fisher matrices
fisher_matrices_all = {key: np.load(f"{fisher_path}{key}_cosmic_shear_fisher_matrix.npy") for key in keys}
fisher_matrices_cosmo = {key: np.load(f"{fisher_path}{key}_cosmic_shear_fisher_matrix_cosmo.npy") for key in keys}

# Calculate the inverse matrices
inverse_fisher_matrices_all = {key: inv(matrix) for key, matrix in fisher_matrices_all.items()}
inverse_fisher_matrices_cosmo = {key: inv(matrix) for key, matrix in fisher_matrices_cosmo.items()}

# Create nested dictionaries for Fisher matrices and their inverses
fisher_matrices = {
    "srd": {
        "all": {"1": fisher_matrices_all["srd_y1"],
                "10": fisher_matrices_all["srd_y10"]},
        "cosmo": {"1": fisher_matrices_cosmo["srd_y1"],
                  "10": fisher_matrices_cosmo["srd_y10"]}
    },
    "srd+lf": {
        "all": {"1": fisher_matrices_all["srd+lf_y1"],
                "10": fisher_matrices_all["srd+lf_y10"]},
        "cosmo": {"1": fisher_matrices_cosmo["srd+lf_y1"],
                  "10": fisher_matrices_cosmo["srd+lf_y10"]},
        "cosmoia": {"1": fisher_matrices_all["srd+lf_y1"][:11, :11],
                    "10": fisher_matrices_all["srd+lf_y10"][:11, :11]}
    },
    "jmas": {
        "all": {"1": fisher_matrices_all["jmas_y1"],
                "10": fisher_matrices_all["jmas_y10"]},
        "cosmo": {"1": fisher_matrices_cosmo["jmas_y1"],
                  "10": fisher_matrices_cosmo["jmas_y10"]},
        "cosmoia": {"1": fisher_matrices_all["jmas_y1"][:11, :11],
                    "10": fisher_matrices_all["jmas_y10"][:11, :11]}
    }
}


# Verify the structure
print(fisher_matrices["srd"].keys())
print(fisher_matrices["srd+lf"].keys())
print(fisher_matrices["jmas"].keys())

np.save(f"{fisher_path}fisher_matrices.npy", fisher_matrices)

# Function to invert the matrices
def invert_matrix_structure(matrix_dict):
    return {k: invert_matrix_structure(v) if isinstance(v, dict) else inv(v) for k, v in matrix_dict.items()}

# Invert all Fisher matrices
inverse_fisher_matrices = invert_matrix_structure(fisher_matrices)

np.save(f"{fisher_path}inverse_fisher_matrices.npy", inverse_fisher_matrices)


srdlf_vals_y1 = parameters["values"]["srd+lf"]["all"]["1"]
srdlf_names_y1 = parameters["names"]["srd+lf"]["all"]["1"]
srdlf_vals_y10 = parameters["values"]["srd+lf"]["all"]["10"]
srdlf_names_y10 = parameters["names"]["srd+lf"]["all"]["10"]
srdlf_labels_y1 = parameters["labels"]["srd+lf"]["all"]["1"]
srdlf_labels_y10 = parameters["labels"]["srd+lf"]["all"]["10"]

srdlf_vals_cosmoia_y1 = parameters["values"]["srd+lf"]["all"]["1"][:11]
srdlf_names_cosmoia_y1 = parameters["names"]["srd+lf"]["all"]["1"][:11]
srdlf_labels_cosmoia_y1 = parameters["labels"]["srd+lf"]["all"]["1"][:11]
srdlf_vals_cosmoia_y10 = parameters["values"]["srd+lf"]["all"]["10"][:11]
srdlf_names_cosmoia_y10 = parameters["names"]["srd+lf"]["all"]["10"][:11]
srdlf_labels_cosmoia_y10 = parameters["labels"]["srd+lf"]["all"]["10"][:11]

jmas_vals_cosmoia_y1 = parameters["values"]["jmas"]["all"]["1"][:11]
jmas_names_cosmoia_y1 = parameters["names"]["jmas"]["all"]["1"][:11]
jmas_labels_cosmoia_y1 = parameters["labels"]["jmas"]["all"]["1"][:11]
jmas_vals_cosmoia_y10 = parameters["values"]["jmas"]["all"]["10"][:11]
jmas_names_cosmoia_y10 = parameters["names"]["jmas"]["all"]["10"][:11]
jmas_labels_cosmoia_y10 = parameters["labels"]["jmas"]["all"]["10"][:11]

parameters["values"]["srd+lf"]["all"].keys()


inverse_fisher_matrices_clipped = {
    "srd": {
        "all": {"1": inv(fisher_matrices["srd"]["all"]["1"]),
                "10": inv(fisher_matrices["srd"]["all"]["10"])},
        "cosmo": {"1":inv(fisher_matrices["srd"]["all"]["1"])[:7, :7],
                  "10": inv(fisher_matrices["srd"]["all"]["10"])[:7, :7]}
    },
    "srd+lf": {
        "all": {"1": inv(fisher_matrices["srd+lf"]["all"]["1"]),
                "10": inv(fisher_matrices["srd+lf"]["all"]["10"])},
        "cosmo": {"1": inv(fisher_matrices["srd+lf"]["all"]["1"])[:7, :7],
                  "10": inv(fisher_matrices["srd+lf"]["all"]["10"])[:7, :7]},
        "cosmoia": {"1": inv(fisher_matrices["srd+lf"]["all"]["1"])[:11, :11],
                    "10": inv(fisher_matrices["srd+lf"]["all"]["10"])[:11, :11]}
    },
    "jmas": {
        "all": {"1": inv(fisher_matrices["jmas"]["all"]["1"]),
                "10": inv(fisher_matrices["jmas"]["all"]["10"])},
        "cosmo": {"1": inv(fisher_matrices["jmas"]["all"]["1"])[:7, :7],
                  "10": inv(fisher_matrices["jmas"]["all"]["10"])[:7, :7]},
        "cosmoia": {"1": inv(fisher_matrices["jmas"]["all"]["1"])[:11, :11],
                    "10": inv(fisher_matrices["jmas"]["all"]["10"])[:11, :11]}
    }
}

np.save("inverse_fisher_matrices_clipped.npy", inverse_fisher_matrices_clipped)

inv_c = inverse_fisher_matrices_clipped

gaussians_fixed = {
    "srd":
        {
            "all":
                {
                    "1": GaussianND(
                        parameters["values"]["srd"]["all"]["1"],
                        inv_c["srd"]["all"]["1"],
                        labels=parameters["labels"]["srd"]["all"]["1"],
                        names=parameters["names"]["srd"]["all"]["1"]),
                    "10": GaussianND(
                        parameters["values"]["srd"]["all"]["10"],
                        inv_c["srd"]["all"]["10"],
                        labels=parameters["labels"]["srd"]["all"]["10"],
                        names=parameters["names"]["srd"]["all"]["10"])
                },
            "cosmo":
                {
                    "1": GaussianND(
                        parameters["values"]["srd"]["cosmo"]["1"],
                        inv_c["srd"]["cosmo"]["1"],
                        labels=parameters["labels"]["srd"]["cosmo"]["1"],
                        names=parameters["names"]["srd"]["cosmo"]["1"]),
                    "10": GaussianND(
                        parameters["values"]["srd"]["cosmo"]["10"],
                        inv_c["srd"]["cosmo"]["10"],
                        labels=parameters["labels"]["srd"]["cosmo"]["10"],
                        names=parameters["names"]["srd"]["cosmo"]["10"], )
                }
        },
    "jmas":
        {
            "all":
                {
                    "1": GaussianND(
                        parameters["values"]["jmas"]["all"]["1"],
                        inv_c["jmas"]["all"]["1"],
                        labels=parameters["labels"]["jmas"]["all"]["1"],
                        names=parameters["names"]["jmas"]["all"]["1"], ),
                    "10": GaussianND(
                        parameters["values"]["jmas"]["all"]["10"],
                        inv_c["jmas"]["all"]["10"],
                        labels=parameters["labels"]["jmas"]["all"]["10"],
                        names=parameters["names"]["jmas"]["all"]["10"], )
                },
            "cosmo":
                {
                    "1": GaussianND(
                        parameters["values"]["jmas"]["cosmo"]["1"],
                        inv_c["jmas"]["cosmo"]["1"],
                        labels=parameters["labels"]["jmas"]["cosmo"]["1"],
                        names=parameters["names"]["jmas"]["cosmo"]["1"], ),
                    "10": GaussianND(
                        parameters["values"]["jmas"]["cosmo"]["10"],
                        inv_c["jmas"]["cosmo"]["10"],
                        labels=parameters["labels"]["jmas"]["cosmo"]["10"],
                        names=parameters["names"]["jmas"]["cosmo"]["10"], )
                },
            "cosmoia":
                {
                    "1": GaussianND(
                        parameters["values"]["jmas"]["cosmoia"]["1"],
                        inv_c["jmas"]["cosmoia"]["1"],
                        labels=parameters["labels"]["jmas"]["cosmoia"]["1"],
                        names=parameters["names"]["jmas"]["cosmoia"]["1"], ),
                    "10": GaussianND(
                        parameters["values"]["jmas"]["cosmoia"]["10"],
                        inv_c["jmas"]["cosmoia"]["10"],
                        labels=parameters["labels"]["jmas"]["cosmoia"]["10"],
                        names=parameters["names"]["jmas"]["cosmoia"]["10"], )
                },
        },
    "srd+lf":
        {
            "all":
                {
                    "1": GaussianND(
                        parameters["values"]["srd+lf"]["all"]["1"],
                        inv_c["srd+lf"]["all"]["1"],
                        labels=parameters["labels"]["srd+lf"]["all"]["1"],
                        names=parameters["names"]["srd+lf"]["all"]["1"], ),
                    "10": GaussianND(
                        parameters["values"]["srd+lf"]["all"]["10"],
                        inv_c["srd+lf"]["all"]["10"],
                        labels=parameters["labels"]["srd+lf"]["all"]["10"],
                        names=parameters["names"]["srd+lf"]["all"]["10"], )
                },
            "cosmo":
                {
                    "1": GaussianND(
                        parameters["values"]["srd+lf"]["cosmo"]["1"],
                        inv_c["srd+lf"]["cosmo"]["1"],
                        labels=parameters["labels"]["srd+lf"]["cosmo"]["1"],
                        names=parameters["names"]["srd+lf"]["cosmo"]["1"], ),
                    "10": GaussianND(
                        parameters["values"]["srd+lf"]["cosmo"]["10"],
                        inv_c["srd+lf"]["cosmo"]["10"],
                        labels=parameters["labels"]["srd+lf"]["cosmo"]["10"],
                        names=parameters["names"]["srd+lf"]["cosmo"]["10"], )
                },
            "cosmoia":
                {
                    "1": GaussianND(
                        parameters["values"]["srd+lf"]["cosmoia"]["1"],
                        inv_c["srd+lf"]["cosmoia"]["1"],
                        labels=parameters["labels"]["srd+lf"]["cosmoia"]["1"],
                        names=parameters["names"]["srd+lf"]["cosmoia"]["1"], ),
                    "10": GaussianND(
                        parameters["values"]["srd+lf"]["cosmoia"]["10"],
                        inv_c["srd+lf"]["cosmoia"]["10"],
                        labels=parameters["labels"]["srd+lf"]["cosmoia"]["10"],
                        names=parameters["names"]["srd+lf"]["cosmoia"]["10"], )
                }
        }

}

np.save("gaussians_clipped.npy", gaussians_fixed)


import numpy as np
from numpy.linalg import inv

# Define the path to the Fisher matrices
fisher_path = "jmas_data/fisher_matrices/"

# List of keys for the Fisher matrices
keys = ["srd_y1", "srd_y10", "jmas_y1", "jmas_y10", "srd+lf_y1", "srd+lf_y10"]

# Load all Fisher matrices
fisher_matrices_all = {key: np.load(f"{fisher_path}{key}_cosmic_shear_fisher_matrix.npy") for key in keys}
fisher_matrices_cosmo = {key: np.load(f"{fisher_path}{key}_cosmic_shear_fisher_matrix_cosmo.npy") for key in keys}

# Calculate the inverse matrices
inverse_fisher_matrices_all = {key: inv(matrix) for key, matrix in fisher_matrices_all.items()}
inverse_fisher_matrices_cosmo = {key: inv(matrix) for key, matrix in fisher_matrices_cosmo.items()}

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
        "cosmoialf": {"1": inv(fisher_matrices["srd"]["all"]["1"]),
                "10": inv(fisher_matrices["srd"]["all"]["10"])},
        "cosmo": {"1":inv(fisher_matrices["srd"]["all"]["1"])[:7, :7],
                  "10": inv(fisher_matrices["srd"]["all"]["10"])[:7, :7]}
    },
    "srd+lf": {
        "cosmoialf": {"1": inv(fisher_matrices["srd+lf"]["all"]["1"]),
                "10": inv(fisher_matrices["srd+lf"]["all"]["10"])},
        "cosmo": {"1": inv(fisher_matrices["srd+lf"]["all"]["1"])[:7, :7],
                  "10": inv(fisher_matrices["srd+lf"]["all"]["10"])[:7, :7]},
        "cosmoia": {"1": inv(fisher_matrices["srd+lf"]["all"]["1"])[:11, :11],
                    "10": inv(fisher_matrices["srd+lf"]["all"]["10"])[:11, :11]}
    },
    "jmas": {
        "cosmoialf": {"1": inv(fisher_matrices["jmas"]["all"]["1"]),
                "10": inv(fisher_matrices["jmas"]["all"]["10"])},
        "cosmo": {"1": inv(fisher_matrices["jmas"]["all"]["1"])[:7, :7],
                  "10": inv(fisher_matrices["jmas"]["all"]["10"])[:7, :7]},
        "cosmoia": {"1": inv(fisher_matrices["jmas"]["all"]["1"])[:11, :11],
                    "10": inv(fisher_matrices["jmas"]["all"]["10"])[:11, :11]}
    }
}
