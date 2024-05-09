import tkinter
import tkinter.messagebox
from typing import Tuple
import customtkinter
from interface.parameter_window import AleParameterWindow
from interface.sidebar_frame import Sidebar_Frame
from interface.analysis_table import AnalysisInfoTable
from interface.output_log_frame import OutputLogFrame



customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("pyALE-GUI")
        self.grid_columnconfigure((1,2), weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.sidebar_frame = Sidebar_Frame(self, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=9, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure((3,5), weight=1)

        self.analysis_info_table = AnalysisInfoTable(self, rows=3, columns=3)
        self.analysis_info_table.grid(row=0, column=1, padx=10, pady=(0, 10), sticky='nsew')

        self.frame_column2 = OutputLogFrame(self, 'Experiment Info')
        self.frame_column2.grid(row=1, column=1, padx=10, pady=(0, 10), sticky='nsew')

        self.frame_column4 = OutputLogFrame(self, 'Analysis Log')
        self.frame_column4.grid(row=0, column=2, rowspan=2, padx=(0,10), pady=(0, 10), sticky='nsew')


if __name__ == '__main__':

    app = App()
    app.mainloop()