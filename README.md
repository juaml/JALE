# ALEpy

A Python package for conducting ALE (Activation Likelihood Estimation) meta-analyses, supporting a range of analysis workflows: standard ALE, probabilistic or cross-validated ALE, standard ALE contrast, and balanced ALE contrast.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Examples](#examples)
- [Background and References](#background-and-references)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the ALE Meta-Analysis Package, run:

```bash
pip install alepy

## Usage

Here’s how to use the project:

ALEpy requires a project folder that contains 3 files:
1. Experiment Data (Author, Subjects, Coordinates, Tags)
2. Analysis Data (Type of ALE, Tags to be included)
3. Yaml config file (specifying project folder path, filenames and ALE parameters)

For example files please check the docs folder.

Running an ALE can be done in two ways:

1. via CLI: python -m alepy /path/to/yaml/file

2. in Python:

```python
from alepy import main

main(yaml_path='/path/to/yaml/file')

## Background and References

This project is based on research by 
- [Author Eickhoff et al., 2012](https://doi.org/10.1016/j.neuroimage.2011.09.017).
- [Author Eickhoff et al., 2016](https://doi.org/10.1016/j.neuroimage.2016.04.072).
- [Author Frahm et al., 2022](https://doi.org/10.1002/hbm.25898).
- [Author Frahm et al., 2023](https://doi.org/10.1016/j.neuroimage.2023.120383).