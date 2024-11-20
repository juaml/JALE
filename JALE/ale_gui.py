import customtkinter

from jale.gui.analysis_table_frame import AnalysisTableFrame
from jale.gui.controller import Controller
from jale.gui.dataset_table_frame import DatasetTableFrame
from jale.gui.output_frame import OutputFrame
from jale.gui.sidebar_frame import Sidebar_Frame

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme(
    "dark-blue"
)  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("pyALE-GUI")
        self.geometry("1200x600")
        self.grid_columnconfigure((1, 2), weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.sidebar_frame = Sidebar_Frame(self, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=9, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure((4), weight=1)

        self.dataset_table_frame = DatasetTableFrame(self)
        self.dataset_table_frame.grid(
            row=0, rowspan=2, column=1, padx=10, pady=(10, 15), sticky="nsew"
        )

        self.analysis_table_frame = AnalysisTableFrame(self)
        self.analysis_table_frame.grid(
            row=2, column=1, padx=10, pady=(0, 10), sticky="nsew"
        )

        self.output_log_frame = OutputFrame(self)
        self.output_log_frame.grid(
            row=0, rowspan=2, column=2, padx=(0, 10), pady=(10, 10), sticky="nsew"
        )

        self.controller = Controller(
            self.sidebar_frame,
            self.analysis_table_frame,
            self.dataset_table_frame,
            self.output_log_frame,
        )
        self.sidebar_frame.set_controller(self.controller)
        self.analysis_table_frame.set_controller(self.controller)
        self.dataset_table_frame.set_controller(self.controller)


if __name__ == "__main__":
    app = App()
    app.mainloop()
