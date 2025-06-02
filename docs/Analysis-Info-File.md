This wiki page explains the format of the analysis info excel file.

Please use the following format / syntax:

![image](https://github.com/user-attachments/assets/a2311b93-8c7a-4351-9769-197fd96ede74)



## 1st column - Type of Analysis:

- 'M': 
  - Standard ALE / Main Effect
- 'P':
  - probabilistic / subsampled / CV'd ALE
  - Need to specify the desired N after the P, e.g. P17
- 'C':
  - Standard contrast (this line and the following are contrasted against each other)
  - If the respective main effects in the two lines [e.g. 3 and 4] are not yet computed, they will be evaluated
- 'B':
  - Balanced contrast
  - Both analysis will be based on the same N
  - Option to specify the desired N after the B, e.g. B17. If no N is provided N will be equal to the smaller analysis N - 2 or the mean 
    between the smaller analysis N and 17 depending on which is smaller.

- 'Cluster': 
  - Hierarchical MA clustering as performed here: [Reference](https://www.nature.com/articles/s41598-022-08909-3).

## 2nd column - Name of the Analysis:
- Ideally avoid spaces and non file-name characters
- When (re-) entering a title as part of a contrast, please keep naming/spelling consistent otherwise the “old” main effect is not recognized
- Names for seperate analyses need to be distinctly different. Otherwise the program will think the analysis was already run and conclude prematurely

## 3rd column onwards - Tags:

Tags are used to run subanalysis of an larger overall dataset (e.g. only include experiments that use a certain version of a task).
The tags are used following a simple logic:

- +ALL -> Include the entire set of experiments in that matfile
- +Tag -> Include only those experiments that have the respective label
- +Tag1 +Tag2 -> Conjunction, includes only experiments that have both labels
- +Tag1 +Tag2 ? -> Logical ‘OR’, includes experiments that have either label
- -Tag -> Exclude experiments that have the respective label


