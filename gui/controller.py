import customtkinter
import pandas as pd
from core.utils.input import load_excel

class Controller:
    def __init__(self, sidebar_frame, analysis_info_table, dataset_table):
        self.sidebar_frame = sidebar_frame
        self.analysis_info_table = analysis_info_table
        self.dataset_table = dataset_table

        # Connect the button event to the controller method
        self.sidebar_frame.import_analysis_file_button.configure(command=self.import_analysis_file)
        self.sidebar_frame.import_dataset_button.configure(command=self.import_dataset)

    def import_analysis_file(self):
        filename = customtkinter.filedialog.askopenfilename()
        if filename:
            self.analysis_info_table.fill_table(filename)
    
    def import_dataset(self):
        filename = customtkinter.filedialog.askopenfilename()
        if filename:
            self.dataset_table.fill_table(filename)

