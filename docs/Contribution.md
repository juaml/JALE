# Contribution File Structure

The `contribution` function generates a `.txt` file for each thresholding technique (vFWE, cFWE, TFCE) containing detailed information about the contribution of individual experiments and tasks to significant clusters identified in the meta-analysis. Below is the detailed structure of the output:

---

## **1. Header Section**

The file begins with a header summarizing the meta-analysis or experiment details:

- **Experiment Name**: The name of the experiment (`exp_name`).
- **Experiment Summary**:
  - The total number of experiments analyzed.
  - The total number of unique subjects across all experiments.
  - The average number of subjects per experiment.

### Example:
```
Starting with MetaAnalysis1!

MetaAnalysis1: 10 experiments; 500 unique subjects (average of 50.0 per experiment)
```

---

## **2. Correction Method-Specific Results**

### **a. Cluster Summary**
- A list of significant clusters detected in the meta-analysis for the given correction method.
- Each cluster includes:
  - **Cluster Index**: The numerical index of the cluster.
  - **Number of Voxels**: Total voxels in the cluster.
  - **Cluster Center Coordinates**: The center of the cluster in MNI space (x/y/z).

### Example:
```
Cluster 1: 25 voxels [Center: -10/20/30]
```

### **b. Experiment Contributions**
- For each experiment contributing significantly to the cluster:
  - **Article Name**: The publication or study name.
  - **Metrics**:
    - **Sum of Activations**: Total activation within the cluster.
    - **Average voxel-wise Contribution**: Average ALE contribution per voxel in the cluster.
    - **Proportional Contribution**: Overall percentagewise contribution, normalized.
    - **Max Contribution**: Maximum contribution to the cluster activation, normalized.
  - **Number of Subjects**: Total subjects in the experiment.

### Example:
```
Study1        0.123   12.3   25.00   50.00   (40)
Study2        0.095   10.5   20.00   45.00   (35)
```

### **c. Task Contributions**
- For each task contributing to the cluster:
  - **Task Name**: Name of the task.
  - **Metrics**:
    - **Sum of Activations**: Total activation within the cluster.
    - **Average voxel-wise Contribution**: Sum of average ALE contribution per voxel in the cluster.
    - **Proportional Contribution**: Overall percentagewise contribution, normalized.

### Example:
```
Task Contributions:
Task1          0.250   15.5   30.00
Task2          0.150   10.0   25.00
```

---

## **3. No Significant Results**

If no significant clusters are found for a correction method, the file includes the following message:
```
No significant clusters in TFCE!
```

---

## **Output File Example**

Below is a complete example of what the output file might look like:

```
Starting with MetaAnalysis1!

MetaAnalysis1: 10 experiments; 500 unique subjects (average of 50.0 per experiment)

Cluster 1: 25 voxels [Center: -10/20/30]

Study1        0.123   12.3   25.00   50.00   (40)
Study2        0.095   10.5   20.00   45.00   (35)

Task Contributions:
Task1          0.250   15.5   30.00
Task2          0.150   10.0   25.00


Cluster 2: 15 voxels [Center: -15/25/35]

Study3        0.105   11.0   22.00   40.00   (30)
Study4        0.080    8.5   18.00   35.00   (25)

Task Contributions:
Task3          0.200   12.5   28.00
Task4          0.100    9.0   24.00
```

---

## **Key Points**

- The output file is organized to provide detailed information for each correction method and its corresponding significant clusters.
- Contributions are broken down into experiments and tasks to analyze their individual impacts.
- If no clusters are detected for a correction method, a note is included to reflect that.

This structure ensures that users can easily interpret which experiments and tasks contribute to specific clusters in a meta-analysis.