from core.utils.input import load_excel, read_experiment_info

class Controller:
    def __init__(self, sidebar_frame, analysis_table_frame, dataset_table_frame, output_log_frame):
         # Frames
         self.sidebar_frame = sidebar_frame
         self.analysis_table_frame = analysis_table_frame
         self.dataset_table_frame = dataset_table_frame
         self.output_log_frame = output_log_frame

         # ALE objects
         self.analysis_df = None
         self.dataset_df = None
         self.task_df = None
         self.parameters = None
         
    # Sidebar Buttons
    def load_analysis_file(self, filename):
        self.analysis_df = load_excel(filename, type='analysis')
        self.analysis_table_frame.fill_table(self.analysis_df)

    def load_dataset_file(self, filename):
        self.dataset_df, self.task_df = read_experiment_info(filename)
        self.dataset_table_frame.fill_table(self.dataset_df)
    
    def open_parameter_window(self):
        self.sidebar_frame.open_parameter_window()

    def handle_parameters(self, parameters):
            self.parameters = parameters

    # Add function that takes manually added analysis logic and sends it to analysis table frame

    def run_analysis(self):
        return
    
    def stop_analysis(self):
        return