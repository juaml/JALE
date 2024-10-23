import sys
from pathlib import Path

import numpy as np
import yaml

from core.analyses.contrast import balanced_contrast, contrast
from core.analyses.main_effect import main_effect
from core.utils.compile_experiments import compile_experiments
from core.utils.contribution import contribution
from core.utils.folder_setup import folder_setup
from core.utils.input import load_excel, read_experiment_info

yaml_path = sys.argv[1]

# Load settings from the YAML file
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

# Accessing specific settings
project_path = config["project"]["path"]
project_path = Path(project_path).resolve()
folder_setup(project_path)

analysis_info_filename = config["project"]["analysis_info"]
experiment_info_filename = config["project"]["experiment_info"]

params = config["parameters"]

exp_all_df, tasks = read_experiment_info(project_path / experiment_info_filename)
analysis_df = load_excel(project_path / analysis_info_filename, type="analysis")

for row_idx in range(analysis_df.shape[0]):
    if type(analysis_df.iloc[row_idx, 0]) is not str:
        continue
    if analysis_df.iloc[row_idx, 0] == "M":  # Main Effect Analysis
        print("Running Main-Effect Analysis")

        meta_name = analysis_df.iloc[row_idx, 1]
        conditions = analysis_df.iloc[row_idx, 2:].dropna().to_list()
        exp_idxs, masks, mask_names = compile_experiments(conditions, tasks)
        exp_df = exp_all_df.loc[exp_idxs].reset_index(drop=True)

        if len(exp_idxs) <= 17:
            print(
                "Analysis contains less than 18 Experiments."
                "Please interprete results carefully!"
            )

        print(
            f"{meta_name} : {len(exp_idxs)} experiments;"
            f"average of {exp_df.Subjects.mean():.2f} subjects per experiment"
        )

        main_effect(
            project_path,
            exp_df,
            meta_name,
            tfce_enabled=params["tfce_enabled"],
            cutoff_predict_enabled=params["cutoff_predict_enabled"],
            bin_steps=params["bin_steps"],
            cluster_forming_threshold=params["cluster_forming_threshold"],
            monte_carlo_iterations=params["monte_carlo_iterations"],
            nprocesses=params["nprocesses"],
        )

        contribution(project_path, exp_df, meta_name, tasks, params["tfce_enabled"])

    # Probabilistic ALE
    if analysis_df.iloc[row_idx, 0][0] == "P":  # type: ignore
        meta_name = analysis_df.iloc[row_idx, 1]
        conditions = analysis_df.iloc[row_idx, 2:].dropna().to_list()
        exp_idxs, _, _ = compile_experiments(conditions, tasks)
        exp_df = exp_all_df.loc[exp_idxs].reset_index(drop=True)
        if len(analysis_df.iloc[row_idx, 0]) > 1:  # type: ignore
            target_n = int(analysis_df.iloc[row_idx, 0][1:])  # type: ignore
            main_effect(
                project_path,
                exp_df,
                meta_name,
                target_n=target_n,
                monte_carlo_iterations=params["monte_carlo_iterations"],
                sample_n=params["subsample_n"],
                nprocesses=params["nprocesses"],
            )
        else:
            print(f"{meta_name}: need to specify subsampling N")
            continue

    if analysis_df.iloc[row_idx, 0] == "C":  # Contrast Analysis
        meta_names = [analysis_df.iloc[row_idx, 1], analysis_df.iloc[row_idx + 1, 1]]
        conditions = [
            analysis_df.iloc[row_idx, 2:].dropna().to_list(),
            analysis_df.iloc[row_idx + 1, 2:].dropna().to_list(),
        ]
        exp_idxs1, masks1, mask_names1 = compile_experiments(conditions[0], tasks)
        exp_idxs2, masks2, mask_names2 = compile_experiments(conditions[1], tasks)
        exp_dfs = [
            exp_all_df.loc[exp_idxs1].reset_index(drop=True),
            exp_all_df.loc[exp_idxs2].reset_index(drop=True),
        ]

        if len(exp_idxs1) < 17 or len(exp_idxs2) < 17:
            print(
                "One of two effects selected for contrast"
                "contains less than 17 Experiments."
                "Please interprete results carefully!"
            )
        for dataset_idx in [0, 1]:
            if not Path(
                project_path
                / f"Results/MainEffect/Full/Volumes/{meta_names[dataset_idx]}_cFWE05.nii"
            ).exists():
                main_effect(
                    project_path,
                    exp_dfs[dataset_idx],
                    meta_names[dataset_idx],
                    tfce_enabled=params["tfce_enabled"],
                    cutoff_predict_enabled=params["cutoff_predict_enabled"],
                    bin_steps=params["bin_steps"],
                    cluster_forming_threshold=params["cluster_forming_threshold"],
                    monte_carlo_iterations=params["monte_carlo_iterations"],
                    nprocesses=params["nprocesses"],
                )
                contribution(
                    project_path,
                    exp_dfs[dataset_idx],
                    meta_names[dataset_idx],
                    tasks,
                    params["tfce_enabled"],
                )

        # Check for overlap in experiments. Need to be removed for contrast to work
        exp_overlap = set(exp_idxs1) & set(exp_idxs2)
        exp_idxs1 = [x for x in exp_idxs1 if x not in exp_overlap]
        exp_idxs2 = [x for x in exp_idxs2 if x not in exp_overlap]

        exp_dfs = [
            exp_all_df.loc[exp_idxs1].reset_index(drop=True),
            exp_all_df.loc[exp_idxs2].reset_index(drop=True),
        ]

        contrast(
            project_path,
            meta_names,
            significance_threshold=params["significance_threshold"],
            null_repeats=params["contrast_permutations"],
            nprocesses=params["nprocesses"],
        )

    if analysis_df.iloc[row_idx, 0][0] == "B":  # Balanced Contrast Analysis:
        meta_names = [analysis_df.iloc[row_idx, 1], analysis_df.iloc[row_idx + 1, 1]]
        conditions = [
            analysis_df.iloc[row_idx, 2:].dropna().to_list(),
            analysis_df.iloc[row_idx + 1, 2:].dropna().to_list(),
        ]
        exp_idxs1, masks1, mask_names1 = compile_experiments(conditions[0], tasks)
        exp_idxs2, masks2, mask_names2 = compile_experiments(conditions[1], tasks)
        exp_dfs = [
            exp_all_df.loc[exp_idxs1].reset_index(drop=True),
            exp_all_df.loc[exp_idxs2].reset_index(drop=True),
        ]

        if len(analysis_df.iloc[row_idx, 0]) > 1:
            target_n = int(analysis_df.iloc[row_idx, 0][1:])
        else:
            n = [len(exp_idxs[0]), len(exp_idxs[1])]
            target_n = int(min(np.floor(np.mean((np.min(n), 17))), np.min(n) - 2))

        for dataset_idx in [0, 1]:
            if not Path(
                project_path
                / f"Results/MainEffect/CV/Volumes/{meta_names[dataset_idx]}_{target_n}.nii"
            ).exists():
                main_effect(
                    project_path,
                    exp_dfs[dataset_idx],
                    meta_names[dataset_idx],
                    target_n=target_n,
                    monte_carlo_iterations=params["monte_carlo_iterations"],
                    sample_n=params["subsample_n"],
                    nprocesses=params["nprocesses"],
                )

        balanced_contrast(
            project_path,
            exp_dfs,
            meta_names,
            target_n,
            difference_iterations=params["difference_iterations"],
            monte_carlo_iterations=params["monte_carlo_iterations"],
            nprocesses=2,
        )
