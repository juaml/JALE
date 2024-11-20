# JALE

![Python Versions](https://img.shields.io/badge/python-%3E%3D3.8-blue)
![Ruff](https://img.shields.io/badge/code%20style-Ruff-blueviolet)
![PyPI Version](https://img.shields.io/pypi/v/jale.svg)
![License](https://img.shields.io/github/license/LenFrahm/JALE.svg)

A Python package for conducting ALE (Activation Likelihood Estimation) meta-analyses, supporting a range of analysis workflows: standard ALE, probabilistic or cross-validated ALE, standard ALE contrast, and balanced ALE contrast.

For information regarding installation, usage and features please visit the [Github Wiki](https://github.com/LenFrahm/JALE/wiki).

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Background and References](#background-and-references)
- [License](#license)

## Installation

To install the ALE Meta-Analysis Package, run:

```bash
pip install jale
```
## Usage

Here’s how to use the project:

JALE requires a project folder that contains 3 files:
1. Experiment Data (Author, Subjects, Coordinates, Tags)
2. Analysis Data (Type of ALE, Tags to be included)
3. Yaml config file (specifying project folder path, filenames and ALE parameters)

For example files please check the wiki.

Running an ALE can be done in two ways:

1. via CLI: 

```bash
python -m jale /path/to/yaml/file
```

2. in Python:

```python
from jale import main

main(yaml_path='/path/to/yaml/file')
```

## Features

- Standard ("Main Effect") ALE
- Probabilistic ("Cross-Validated") ALE
- Standard ALE Contrast
- Balanced ALE Contrast

## Background and References

This project is based on research by 
- [Eickhoff et al., 2012](https://doi.org/10.1016/j.neuroimage.2011.09.017).
- [Eickhoff et al., 2016](https://doi.org/10.1016/j.neuroimage.2016.04.072).
- [Frahm et al., 2022](https://doi.org/10.1002/hbm.25898).
- [Frahm et al., 2023](https://doi.org/10.1016/j.neuroimage.2023.120383).
