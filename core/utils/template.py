import nibabel as nb
import numpy as np
from pathlib import Path

# Module loads grey matter mask and saves important characteristics like shape, etc.
module_path = Path(__file__).resolve().parents[2]
template = nb.load(module_path / "/mask/Grey10.nii")

data = template.get_fdata()
shape = data.shape
pad_shape = np.array([value+30 for value in shape])

prior = np.zeros(shape, dtype=bool)
prior[data > 0.1] = 1
sample_space = np.array(np.where(prior == 1))

affine = template.affine
