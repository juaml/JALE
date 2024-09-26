import nibabel as nb
import numpy as np
from pathlib import Path

# Module loads grey matter mask and saves important characteristics like shape, etc.
module_path = Path(__file__).resolve().parents[2]
template = nb.load(module_path / "/mask/Grey10.nii")

data = template.get_fdata()
BRAIN_ARRAY_SHAPE = data.shape
PAD_SHAPE = np.array([value+30 for value in BRAIN_ARRAY_SHAPE])

GM_PRIOR = np.zeros(BRAIN_ARRAY_SHAPE, dtype=bool)
GM_PRIOR[data > 0.1] = 1
GM_SAMPLE_SPACE = np.array(np.where(GM_PRIOR == 1))

MNI_AFFINE = template.affine
