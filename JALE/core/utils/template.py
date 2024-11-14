"""
Module for loading and storing key constants for the MNI template and grey matter mask.
"""

from importlib import resources

import nibabel as nb
import numpy as np

# Define the path to the template file

with resources.path("jale.assets.mask", "Grey10.nii") as mask_file_path:
    template = nb.loadsave.load(mask_file_path)

# Extract template data
data = template.get_fdata()

# Constants derived from the template data
BRAIN_ARRAY_SHAPE = data.shape  # Shape of the brain array in the template
PAD_SHAPE = np.array(
    [value + 30 for value in BRAIN_ARRAY_SHAPE]
)  # Padded shape for specific operations

# Grey matter mask as a binary array, used for sampling
GM_PRIOR = np.zeros(BRAIN_ARRAY_SHAPE, dtype=bool)
GM_PRIOR[data > 0.1] = 1  # Thresholding to create the grey matter mask

# Sampling space for grey matter regions
GM_SAMPLE_SPACE = np.array(np.where(GM_PRIOR == 1))

# Affine transformation matrix for MNI space
MNI_AFFINE = template.affine
