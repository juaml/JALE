# How to Create a YAML Configuration File for Your Project

This guide provides step-by-step instructions on how to create a YAML file for configuring your ALE analysis. YAML (YAML Ain't Markup Language) is a human-readable format commonly used for configuration files.

---

## Step 1: Open a Text Editor

You can use any text editor to create a YAML file. Popular options include:

- **Visual Studio Code**
- **Notepad++**
- **Sublime Text**
- **nano** or **vim** (for terminal users)

---

## Step 2: Save the File with a `.yaml` or `.yml` Extension

When you create the file, save it with an appropriate name, such as `config.yaml`. Ensure the file extension is `.yaml` or `.yml`.

---

## Step 3: Copy the YAML Template Below

Below is the YAML template for your project. Copy this content into your file:

```yaml
# Project Folder
project:
  analysis_info: "analysis_info.xlsx"
  experiment_info: "experiment_info.xlsx"


# ALE Parameters
# DO NOT CHANGE THESE IF YOU ARE NOT AN ALE EXPERT!
parameters:
  
  # If enabled pools multiple experiments from same paper into one experiment;
  pool_experiments: False

  # TFCE is a method for multiple comparison correction (Frahm et al., 2022)
  tfce_enabled: True

  # if enabled ALE map (and therefore all following maps) will be masked by ICBM 10% GM mask
  gm_masking: True

  # Size of bins used in MA histogram
  # Default: 0.0001
  bin_steps: 0.0001 
  
  # If enabled uses xgboost models to predict cutoffs instead of monte carlo simulation (Frahm et al., 2024)
  cutoff_predict_enabled: True

  # P-value required for significance
  # !! ONLY IF CUTOFF PREDICT IS DISABLED, OTHERWISE WON'T HAVE EFFECT !!
  # Default: 0.05
  significance_threshold: 0.05

  # Preliminary cluster forming threshold used in cluster-level family wise error correction
  # !! ONLY IF CUTOFF PREDICT IS DISABLED, OTHERWISE WON'T HAVE EFFECT !!
  # Default: 0.001
  cluster_forming_threshold: 0.001

  # Iterations used for monte-carlo based multiple comparison correction 
  # !! ONLY IF CUTOFF PREDICT IS DISABLED, OTHERWISE WON'T HAVE EFFECT !!
  # Default: 5000
  monte_carlo_iterations: 5000

  # N subsamples calculated for probabilistic ALE algorithm
  # Default: 2500
  subsample_n: 2500

  # Iterations used in classic contrast algorithm
  # Default: 10000
  contrast_permutations: 10000

  # Correction Method on which to base contrast algorithm
  # Default: "cFWE"; Options: "cFWE", "vFWE", "tfce"
  contrast_correction_method: "cFWE"

  # Sub-Iterations used in balanced contrast algorithm
  # Default: 1000
  difference_iterations: 1000

  # Number of parallel processes used for many different steps in the ALE algorith - maximum depends on your machine
  # Default: 2
  nprocesses: 2 


# DO NOT CHANGE THESE IF YOU ARE NOT AN ALE EXPERT

# MA_Clustering Parameters
# DO NOT CHANGE THESE IF YOU ARE NOT AN ALE EXPERT!
clustering_parameters:
max_clusters: 10                  # Default: 10

subsample_fraction: 0.9           # Default: 0.9

sampling_iterations: 1000          # Default: 1000

null_iterations: 5000             # Default: 5000

correlation_type: "spearman"      # Default: "spearman"; Options: "spearman", "pearson"

clustering_method: "hierarchical" # Default: "hierarchical"; Options: "hierarchical", "kmedoids"

linkage_method: "complete"        # Default: "complete"; Options: "complete", "average"

use_pooled_std: False             # Default: False

# DO NOT CHANGE THESE IF YOU ARE NOT AN ALE EXPERT
```

## Step 4: Save the File

After pasting the content change the names of your experiment and analysis info file and save the file with the name config.yaml.