# **Probablistic ALE: Checking Robustness of Main Effect Results**

A **jackknife analysis** (a type of Probabilistic ALE) involves systematically repeating an ALE meta-analysis multiple times, leaving out one experiment in each iteration. If there are \( N \) experiments in the dataset, \( N \) ALE analyses are conducted, each excluding a single experiment. This approach provides insight into the robustness and reliability of the main effect ALE results.

---

## **Steps in a Jackknife Analysis**

1. **Initial ALE Analysis**:
   - Conduct the main effect ALE meta-analysis using all \( N \) experiments to identify significant activation patterns.

2. **Iterative Exclusion**:
   - For each experiment \( i \) (\( i = 1, 2, ..., N \)):
     - Exclude the \( i \)-th experiment from the dataset.
     - Perform the ALE meta-analysis on the remaining \( N-1 \) experiments.

3. **Compare Results**:
   - Evaluate the extent to which the identified activation patterns (clusters) from the full dataset are preserved across the \( N \) reduced analyses.

---

## **Benefits of Jackknife Analysis for Robustness Checks**

1. **Assessing Consistency**:
   - Jackknife analysis reveals whether significant clusters identified in the main effect are consistently observed when each experiment is excluded.
   - Clusters that remain significant across all \( N \) analyses are considered robust and less influenced by individual experiments.

2. **Detecting Outliers**:
   - If excluding a particular experiment causes a dramatic change in the results (e.g., disappearance or emergence of significant clusters), it may indicate that the experiment is an outlier or disproportionately driving the results.

3. **Quantifying Variability**:
   - By comparing voxel-wise ALE values across all \( N \) analyses, it is possible to quantify the variability in activation patterns.
   - Regions with high variability may require careful interpretation, as their significance depends heavily on the included experiments.

4. **Identifying Core Clusters**:
   - Clusters that remain significant across all \( N \) analyses (or most of them) can be considered core activation regions, indicating strong convergence across studies.

5. **Guiding Interpretations**:
   - The analysis provides a robustness metric that helps interpret results conservatively. Regions heavily influenced by individual studies may be less reliable.

---

## **Limitations of Jackknife Analysis**

1. **Computational Cost**:
   - Performing \( N \) ALE analyses can be computationally intensive, particularly for large datasets.

2. **Sensitivity to Dataset Size**:
   - Smaller datasets (\( N \)) may yield less stable jackknife results due to higher sensitivity to individual experiments.

3. **Does Not Account for Systematic Bias**:
   - While the jackknife can identify experiments with disproportionate influence, it does not address systematic biases across experiments (e.g., methodological differences).

---

## **Interpretation of Results**

1. **High Robustness**:
   - If significant clusters persist across all \( N \) jackknife analyses, the main effect ALE results are likely robust and reliable.
   - These regions can be confidently interpreted as areas of convergence.

2. **Low Robustness**:
   - If clusters appear or disappear frequently depending on the excluded experiment, it indicates low robustness. This suggests that some clusters are highly dependent on specific studies.

3. **Outlier Detection**:
   - Identifying experiments whose exclusion dramatically changes results can guide further scrutiny of those studies (e.g., methodological differences, small sample size).

---

## **Conclusion**

A **jackknife analysis** is a powerful tool to validate the robustness of ALE meta-analysis results. By systematically assessing the impact of each experiment on the overall findings, this method helps ensure that significant activation patterns are not overly influenced by individual studies, enhancing the reliability of conclusions. It is particularly valuable in meta-analyses involving heterogeneous datasets, where the robustness of results is critical for accurate interpretation.