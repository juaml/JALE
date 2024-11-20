import customtkinter

from jale.gui.add_analysis_window import AddAnalysisWindow
from jale.gui.parameter_window import ParameterWarningWindow, ParameterWindow


class Sidebar_Frame(customtkinter.CTkFrame):
    def __init__(self, master, corner_radius: int = 0):
        super().__init__(master, corner_radius=corner_radius)
        self.add_analysis_window = None
        self.parameter_window = None
        self.parameter_warning_window = None

        self.import_dataset_button = customtkinter.CTkButton(
            master=self,
            text="Import Dataset",
            command=self.import_dataset_file_button_event,
        )
        self.import_dataset_button.grid(row=0, column=0, padx=20, pady=(20, 20))

        self.add_analysis_button = customtkinter.CTkButton(
            master=self,
            text="Add Analysis",
            command=self.add_analysis_button_event,
            state="disabled",
        )
        self.add_analysis_button.grid(row=1, column=0, padx=20, pady=(20, 5))

        self.save_analysis_button = customtkinter.CTkButton(
            master=self,
            text="Save Analysis",
            command=self.save_analysis_button_event,
            state="disabled",
        )
        self.save_analysis_button.grid(row=2, column=0, padx=20, pady=(5, 5))

        self.reset_table_button = customtkinter.CTkButton(
            self,
            text="Reset Analysis",
            command=self.reset_table_button_event,
            state="disabled",
        )
        self.reset_table_button.grid(row=3, column=0, padx=20, pady=(5, 5))

        self.ale_parameters_button = customtkinter.CTkButton(
            master=self, text="ALE Parameters", command=self.ale_parameters_button_event
        )
        self.ale_parameters_button.grid(row=4, column=0, padx=20, pady=10)

        self.run_analysis_button = customtkinter.CTkButton(
            master=self,
            text="Run Analysis",
            fg_color="green4",
            hover_color="dark green",
            command=self.run_analysis_button_event,
        )
        self.run_analysis_button.grid(row=6, column=0, padx=20, pady=10)

        self.stop_analysis_button = customtkinter.CTkButton(
            master=self,
            text="Stop Analysis",
            fg_color="red3",
            hover_color="red4",
            command=self.stop_analysis_button_event,
        )
        self.stop_analysis_button.grid(row=7, column=0, padx=20, pady=10)

        self.appearance_mode_label = customtkinter.CTkLabel(
            master=self, text="Appearance Mode:", anchor="w"
        )
        self.appearance_mode_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(
            master=self,
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode_event,
        )
        self.appearance_mode_optionemenu.set("Dark")
        self.appearance_mode_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 10))

        self.scaling_label = customtkinter.CTkLabel(
            master=self, text="UI Scaling:", anchor="w"
        )
        self.scaling_label.grid(row=10, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(
            master=self,
            values=["80%", "90%", "100%", "110%", "120%"],
            command=self.change_scaling_event,
        )
        self.scaling_optionemenu.set("100%")
        self.scaling_optionemenu.grid(row=11, column=0, padx=20, pady=(10, 20))

    def set_controller(self, controller):
        self.controller = controller

    def import_dataset_file_button_event(self):
        filename = customtkinter.filedialog.askopenfilename()
        if filename:
            self.controller.load_dataset_file(filename)
            self.add_analysis_button.configure(state="normal")
            self.save_analysis_button.configure(state="normal")
            self.reset_table_button.configure(state="normal")

    def add_analysis_button_event(self):
        if (
            self.add_analysis_window is None
            or not self.add_analysis_window.winfo_exists()
        ):
            self.add_analysis_window = AddAnalysisWindow(self, self.controller)
        else:
            self.add_analysis_window.focus()

    def save_analysis_button_event(self):
        pass

    def reset_table_button_event(self):
        self.controller.analysis_df = None
        self.controller.reset_analysis_table()

    def ale_parameters_button_event(self):
        if self.parameter_window is None or not self.parameter_window.winfo_exists():
            if (
                self.parameter_warning_window is None
                or not self.parameter_warning_window.winfo_exists()
            ):
                self.parameter_warning_window = ParameterWarningWindow(
                    self, self.controller
                )
        else:
            self.parameter_warning_window.focus()

    def open_parameter_window(self):
        self.parameter_window = ParameterWindow(self, self.controller)

    def run_analysis_button_event(self):
        self.controller.run_analysis()
        return

    def stop_analysis_button_event(self):
        self.controller.stop_analysis()
        return

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
