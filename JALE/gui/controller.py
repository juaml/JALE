import pandas as pd

from jale.core.utils.input import load_excel, read_experiment_info


class Controller:
    def __init__(
        self, sidebar_frame, analysis_table_frame, dataset_table_frame, output_log_frame
    ):
        # Frames
        self.sidebar_frame = sidebar_frame
        self.analysis_table_frame = analysis_table_frame
        self.dataset_table_frame = dataset_table_frame
        self.output_log_frame = output_log_frame

        # ALE objects
        self.analysis_df = pd.DataFrame(
            columns=["analysis_type", "analysis_name", "group1_logic", "group2_logic"]
        )
        self.dataset_df = None
        self.task_df = None
        self.parameters = None

    # Sidebar Buttons
    def load_dataset_file(self, filename):
        self.dataset_df, self.task_df = read_experiment_info(filename)
        self.dataset_table_frame.fill_table(self.dataset_df)

    def open_parameter_window(self):
        self.sidebar_frame.open_parameter_window()

    def get_ale_parameters(self, ale_parameters):
        self.parameters = ale_parameters

    def import_analysis_file(self, filename):
        self.analysis_df = load_excel(filename, type="analysis")
        self.analysis_df = self.analysis_table_frame.format_imported_analysis_file(
            self.analysis_df
        )
        self.analysis_table_frame.fill_table(self.analysis_df)

    def get_analysis_parameters(self, analysis_parameters):
        if "group2_logic" not in analysis_parameters:
            analysis_parameters["group2_logic"] = "----"
        new_row = pd.DataFrame([analysis_parameters])
        self.analysis_df = pd.concat([self.analysis_df, new_row], ignore_index=True)
        self.analysis_table_frame.fill_table(self.analysis_df)

    def reset_analysis_table(self):
        self.analysis_table_frame.reset_table()

    def run_analysis(self):
        return

    def stop_analysis(self):
        return
