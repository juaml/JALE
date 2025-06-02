# Configuration Parameters for ALE

This page provides a detailed explanation of the parameters available in the `config.yml` file. These settings control various aspects of the ALE analysis workflow. **Please note:** Many of these parameters are intended for advanced users. **DO NOT CHANGE THESE IF YOU ARE NOT AN ALE EXPERT!**

---

## Project Folder

The **Project Folder** section specifies the file paths for key project documents:

- **`analysis_info`**  
  - **Default:** `"analysis_info.xlsx"`  
  - **Description:** Excel file containing analysis-related information.

- **`experiment_info`**  
  - **Default:** `"experiment_info.xlsx"`  
  - **Description:** Excel file containing details about the experiments.

---

## ALE Parameters

These parameters control the main ALE analysis settings:

- **`pool_experiments`**  
  - **Default:** `True`  
  - **Description:** When enabled, multiple experiments from the same paper are pooled into a single experiment. This helps in cases where individual experiments are not statistically independent.

- **`tfce_enabled`**  
  - **Default:** `True`  
  - **Description:** Enables TFCE (Threshold-Free Cluster Enhancement) for multiple comparison correction. (See Frahm et al., 2022 for details.)

- **`gm_masking`**  
  - **Default:** `True`  
  - **Description:** If enabled, the ALE map—and all subsequent maps—will be masked by the ICBM 10% GM mask to restrict the analysis to gray matter regions.

- **`bin_steps`**  
  - **Default:** `0.0001`  
  - **Description:** Defines the size of the bins used in the Modeled Activation (MA) histogram. Smaller steps provide a finer resolution of the histogram.

- **`cutoff_predict_enabled`**  
  - **Default:** `True`  
  - **Description:** When enabled, xgboost models are used to predict cutoff values instead of relying on monte carlo simulation. (Refer to Frahm et al., 2024.)

- **`significance_threshold`**  
  - **Default:** `0.05`  
  - **Description:** The p-value required for significance. **Important:** This setting only applies if `cutoff_predict_enabled` is disabled.

- **`cluster_forming_threshold`**  
  - **Default:** `0.001`  
  - **Description:** Sets the preliminary cluster forming threshold for cluster-level family-wise error correction. **Important:** This is only effective if `cutoff_predict_enabled` is disabled.

- **`monte_carlo_iterations`**  
  - **Default:** `5000`  
  - **Description:** Number of iterations used for monte-carlo based multiple comparison correction. **Note:** Only applicable when `cutoff_predict_enabled` is disabled.

- **`subsample_n`**  
  - **Default:** `2500`  
  - **Description:** The number of subsamples calculated for the probabilistic ALE algorithm.

- **`contrast_permutations`**  
  - **Default:** `10000`  
  - **Description:** Specifies the number of iterations used in the classic contrast algorithm.

- **`contrast_correction_method`**  
  - **Default:** `"cFWE"`  
  - **Options:** `"cFWE"`, `"vFWE"`, `"tfce"`  
  - **Description:** Determines the correction method for the contrast algorithm.  
    - `"cFWE"`: Cluster-level family-wise error correction (default).  
    - `"vFWE"`: Voxel-level family-wise error correction.  
    - `"tfce"`: Threshold-Free Cluster Enhancement correction.

- **`difference_iterations`**  
  - **Default:** `1000`  
  - **Description:** Number of sub-iterations used in the balanced contrast algorithm.

- **`nprocesses`**  
  - **Default:** `2`  
  - **Description:** Sets the number of parallel processes for various steps in the ALE algorithm. Adjust this number based on the capabilities of your machine.

---

## MA_Clustering Parameters

These parameters control the settings for the Modeled Activation (MA) clustering process. **DO NOT CHANGE THESE IF YOU ARE NOT AN ALE EXPERT!**

- **`max_clusters`**  
  - **Default:** `10`  
  - **Description:** The maximum number of clusters allowed in the analysis.

- **`subsample_fraction`**  
  - **Default:** `0.9`  
  - **Description:** The fraction of data used in each subsampling iteration during the clustering process.

- **`sampling_iterations`**  
  - **Default:** `1000`  
  - **Description:** Number of iterations for subsampling in the clustering algorithm.

- **`null_iterations`**  
  - **Default:** `5000`  
  - **Description:** Number of iterations used to generate the null distribution for statistical testing.

- **`correlation_type`**  
  - **Default:** `"spearman"`  
  - **Options:** `"spearman"`, `"pearson"`  
  - **Description:** Specifies the type of correlation coefficient used to assess similarity between clusters.

- **`clustering_method`**  
  - **Default:** `"hierarchical"`  
  - **Options:** `"hierarchical"`, `"kmeans"`  
  - **Description:** Determines the clustering method. Hierarchical clustering is the default.

- **`linkage_method`**  
  - **Default:** `"complete"`  
  - **Options:** `"complete"`, `"average"`, `"ward"`  
  - **Description:** Defines the linkage method for hierarchical clustering.

---

## Important Notes

- **Modifying Parameters:** Changes to the `config.yml` file can significantly affect the outcome of your analyses. Only adjust these parameters if you fully understand their implications.
- **Further Reading:** For more detailed explanations and the latest updates, please refer to the ALE documentation and the cited publications (e.g., Frahm et al., 2022; Frahm et al., 2024).