import customtkinter


class Sidebar_Frame(customtkinter.CTkFrame):
    def __init__(self, master, corner_radius: int = 0):
        super().__init__(master, corner_radius=corner_radius)
        self.add_analysis_window = None
        self.parameter_window = None
        self.parameter_warning_window = None

        self.select_config_file_button = customtkinter.CTkButton(
            master=self,
            text="Select ALE Config",
            command=self.select_config_file_button_event,
        )
        self.select_config_file_button.grid(row=0, column=0, padx=20, pady=(20, 20))

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

    def select_config_file_button_event(self):
        self.controller.config_to_gui()
        return

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
