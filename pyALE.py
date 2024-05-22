import tkinter
import tkinter.messagebox
from typing import Tuple
import customtkinter
from gui.sidebar_frame import Sidebar_Frame
from gui.analysis_table import AnalysisInfoTable
from gui.dataset_table import DatasetTable
from gui.output_log_frame import OutputLogFrame
from gui.controller import Controller



customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("pyALE-GUI")
        self.geometry("1800x800")
        self.grid_columnconfigure((1,2), weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.sidebar_frame = Sidebar_Frame(self, controller=self, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=9, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure((3), weight=1)

        self.analysis_info_table = AnalysisInfoTable(self)
        self.analysis_info_table.grid(row=0, column=1, padx=10, pady=(10, 10), sticky='nsew')

        self.dataset_table = DatasetTable(self)
        self.dataset_table.grid(row=1, rowspan=2, column=1, padx=10, pady=(0, 10), sticky='nsew')

        self.analysis_log = OutputLogFrame(self, 'Analysis Log')
        self.analysis_log.grid(row=0, column=2, rowspan=2, padx=(0,10), pady=(10, 10), sticky='nsew')

        self.controller = Controller(self.sidebar_frame, self.analysis_info_table, self.dataset_table)



if __name__ == '__main__':

    app = App()
    app.mainloop()