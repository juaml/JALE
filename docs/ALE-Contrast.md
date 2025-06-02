# Explanation of the Standard and Balanced ALE Contrast Algorithms

---

## **Introduction**
Activation Likelihood Estimation (ALE) meta-analysis is a statistical method used to determine convergence of brain activity across neuroimaging studies. It is often employed to contrast two datasets (e.g., groups or tasks). Two main approaches for such contrasts include the **Standard ALE Contrast Algorithm** and the **Balanced ALE Contrast Algorithm**. These methods are designed to compare voxel-wise differences in convergence, but they differ in how they handle datasets with unequal sizes.

---

## **Standard ALE Contrast Algorithm**

The **Standard ALE Contrast Algorithm** compares two datasets as they are, regardless of size differences. This approach follows these steps:

1. **Compute ALE difference Map**:
   - Compute ALE values for each voxel in both datasets, which reflect the likelihood of brain activity convergence. Subtract the ALE maps from each other

2. **Generate Null Distribution**:
   - A null distribution of voxel-wise difference values is created by repeating the following steps 5000 (default) times:
     - randomly permute experiments between the two datasets
     - compute ALE map for the permuted datasets
     - compute permuted ALE difference
     - store voxel-wise difference values

3. **Statistical Testing**:
   - Voxel-wise differences between the ALE maps of the two datasets are compared to the null distribution to identify significant effects.

### **Key Characteristics**:
- **Strength**: No false positives are introduced; the method is statistically robust. Quick to calculate.
- **Limitation**: Larger datasets dominate the contrast, potentially masking subtle differences in smaller datasets.

---

## **Balanced ALE Contrast Algorithm**

The **Balanced ALE Contrast Algorithm** addresses the size imbalance issue by using subsampling to ensure equal comparison. Its steps include:

1. **Subsampling**:
   - Create multiple same-sized subsets from both datasets.
   - For example, randomly select 30 experiments from the larger dataset (150 experiments) to match the smaller dataset (30 experiments).

2. **Compute Subsampled ALE Maps**:
   - Compute ALE maps for the subsamples and calculate the differences.

3. **Generate Null Distribution**:
   - A null distribution is created from null datasets of the same size as the subsamples.
   - For each pair of null datasets we calculate subsample ALE difference maps and average.

4. **Averaging Results**:
   - Average the differences across all subsamples and compare to null distribution to determine significant effects.

### **Key Characteristics**:
- **Strength**: Reduces the bias introduced by dataset size differences, making it more sensitive to subtle effects.
- **Limitation**: Computationally way more expensive due to repeated subsampling.

--- 

## **Simulation Study**
A simulation study compared these algorithms using datasets of 30 and 150 experiments with varying levels of spatial convergence (e.g., the number of experiments with similar peak coordinates). Key findings include:

1. **Standard Contrast**:
   - Showed robustness (no false positives) but was dominated by the larger dataset.
   - Detected fewer significant effects in smaller datasets.

2. **Balanced Contrast**:
   - More sensitive to small differences, even in imbalanced datasets.
   - Detected significant effects at the "true location" more frequently.

---

## **Conclusion**
- **Standard Contrast**: Valid and robust but less sensitive in detecting small differences. Useful when dataset sizes are comparable.
- **Balanced Contrast**: Better suited for cases where dataset sizes differ significantly and small differences are expected.

For meta-analyses involving large size differences, the **Balanced Contrast Algorithm** is recommended to minimize bias and enhance sensitivity.