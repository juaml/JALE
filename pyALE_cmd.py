import sys
import yaml
from pathlib import Path
from core.utils.input import read_experiment_info, load_excel
from core.utils.compile_experiments import compile_experiments
from core.utils.folder_setup import folder_setup
from core.analyses.main_effect import main_effect
from core.utils.contribution import contribution

yaml_path = sys.argv[1]

# Load settings from the YAML file
with open(yaml_path, 'r') as file:
    config = yaml.safe_load(file)

# Accessing specific settings
project_path = config['project']['path']
project_path = Path(project_path).resolve()
analysis_info_filename = config['project']['analysis_info']
experiment_info_filename = config['project']['experiment_info']

params = config['parameters']

exp_all_df, tasks = read_experiment_info(project_path /
                                         experiment_info_filename)
analysis_df = load_excel(project_path /
                         analysis_info_filename, type="analysis")

for row_idx in range(analysis_df.shape[0]):
    if type(analysis_df.iloc[row_idx, 0]) is not str:
        continue
    if analysis_df.iloc[row_idx, 0] == "M":
        # Main Effect Analysis
        print("Running Main-Effect Analysis")
        if not Path(project_path / "Results/MainEffect/Full").exists():
            folder_setup(project_path, "MainEffect_Full")
        exp_name = analysis_df.iloc[row_idx, 1]
        conditions = analysis_df.iloc[row_idx, 2:].dropna().to_list()
        exp_idxs, masks, mask_names = compile_experiments(conditions, tasks)
        exp_df = exp_all_df.loc[exp_idxs].reset_index(drop=True)
        if len(exp_idxs) <= 17:
            print("Analysis contains less than 18 Experiments."
                  "Please interprete results carefully!")
        print(f"{exp_name} : {len(exp_idxs)} experiments;"
              f"average of {exp_df.Subjects.mean():.2f} subjects per experiment")
        main_effect(project_path,
                    exp_df,
                    exp_name,
                    tfce_enabled=params['tfce_enabled'],
                    cutoff_predict_enabled=params['cutoff_predict_enabled'],
                    bin_steps=params['bin_steps'],
                    cluster_forming_threshold=params['cluster_forming_threshold'],
                    monte_carlo_iterations=params['monte_carlo_iterations'],
                    nprocesses=params['nprocesses'])
        contribution(project_path,
                     exp_df,
                     exp_name,
                     tasks,
                     params['tfce_enabled'])

    if analysis_df.iloc[row_idx, 0][0] == "P":  # Probabilistic ALE
        if not Path(project_path /
                    "Results/MainEffect/CV").exists():
            folder_setup(project_path, "MainEffect_CV")
        exp_name = analysis_df.iloc[row_idx, 1]
        conditions = analysis_df.iloc[row_idx, 2:].dropna().to_list()
        exp_idxs, _, _ = compile_experiments(conditions, tasks)
        exp_df = exp_all_df.loc[exp_idxs].reset_index(drop=True)
        if len(analysis_df.iloc[row_idx, 0]) > 1:
            target_n = int(analysis_df.iloc[row_idx, 0][1:])
            main_effect(project_path,
                        exp_df,
                        exp_name,
                        target_n=target_n,
                        monte_carlo_iterations=params['monte_carlo_iterations'],
                        sample_n=params['subsample_n'],
                        nprocesses=params['nprocesses'])
        else:
            print(f"{exp_name}: need to specify subsampling N")
            continue
