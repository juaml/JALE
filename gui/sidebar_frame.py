import customtkinter
from gui.parameter_window import ParameterWindow
from gui.fonts.fonts import get_font

class Sidebar_Frame(customtkinter.CTkFrame):
    def __init__(self, master, controller, corner_radius: int = 0):
        super().__init__(master, corner_radius=corner_radius)
        self.controller = controller
        self.parameter_window = None

        self.logo_label = customtkinter.CTkLabel(master=self, text="pyALE", anchor="w", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=(20,0), pady=(20, 10), sticky='ew')


        self.import_analysis_file_button = customtkinter.CTkButton(master=self, text='Import Analysis File')
        self.import_analysis_file_button.grid(row=1, column=0, padx=20, pady=10)

        self.import_dataset_button = customtkinter.CTkButton(master=self, text='Import Dataset')
        self.import_dataset_button.grid(row=2, column=0, padx=20, pady=10)

        self.ale_parameters_button = customtkinter.CTkButton(master=self, text='ALE Parameters', command=self.ale_parameters_button_event)
        self.ale_parameters_button.grid(row=5, column=0, padx=20, pady=10)

        custom_font = get_font(family='Cascadia Code', size=12)
        self.parameters_text = customtkinter.CTkLabel(master=self, font=custom_font, text='Default parameter\nvalues are set\nautomatically.\nChanging parameters\nonly advised\nfor experts.', anchor='w', justify='left')
        self.parameters_text.grid(row=4, column=0, padx=0, pady=10)

        self.run_analysis_button = customtkinter.CTkButton(master=self, text='Run Analysis', fg_color='green', command=self.run_analysis_button_event)
        self.run_analysis_button.grid(row=6, column=0, padx=20, pady=10,)

        self.stop_analysis_button = customtkinter.CTkButton(master=self, text='Stop Analysis', fg_color='red', command=self.stop_analysis_button_event)
        self.stop_analysis_button.grid(row=7, column=0, padx=20, pady=10,)

        self.appearance_mode_label = customtkinter.CTkLabel(master=self, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(master=self, values=["Light", "Dark", "System"],
                                                                        command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.set("Dark")
        self.appearance_mode_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(master=self, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=10, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(master=self, values=["80%", "90%", "100%", "110%", "120%"],
                                                                command=self.change_scaling_event)
        self.scaling_optionemenu.set("100%")
        self.scaling_optionemenu.grid(row=11, column=0, padx=20, pady=(10, 20))

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def ale_parameters_button_event(self):
        if self.parameter_window == None or not self.parameter_window.winfo_exists():
            self.parameter_window = ParameterWindow(self)
        else:
            self.parameter_window.focus()

    def run_analysis_button_event(self):
        return
    
    def stop_analysis_button_event(self):
        return