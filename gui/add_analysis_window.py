import customtkinter
import tkinter

class AddAnalysisWindow(customtkinter.CTkToplevel):
    def __init__(self, master, task_df):
        super().__init__(master)
        self.task_df = task_df
        self.title("Specify ALE Analyses")
        self.grid_columnconfigure(0, weight=1)

        self.analysis_type_label = customtkinter.CTkLabel(self, text="Analysis Type")
        self.analysis_type_label.grid(row=1, column=0, padx=10, pady=(10, 0), sticky='w')
        self.analysis_type = customtkinter.CTkOptionMenu(self, values=["Main Effect", "Intrastudy probablistic ALE", "Standard Contrast", "Balanced Contrast"],
                                                         command=self.update_group2_state)
        self.analysis_type.grid(row=2, column=0, padx=10, pady=10, sticky='w')

        self.analysis_name_label = customtkinter.CTkLabel(self, text="Analysis Name")
        self.analysis_name_label.grid(row=1, column=1, padx=10, pady=(10, 0), sticky='w')
        self.analysis_name = customtkinter.CTkEntry(self)
        self.analysis_name.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        # Experiment Group1
        self.tag1_label = customtkinter.CTkLabel(self, text="Tag1")
        self.tag1_label.grid(row=1, column=2, padx=10, pady=(10, 0), sticky='w')
        self.tag1 = customtkinter.CTkOptionMenu(self, values=self.task_df.Name.values, command=self.tag_all_update_states)
        self.tag1.grid(row=2, column=2, padx=10, pady=10, sticky='w')

        self.logic1 = customtkinter.CTkOptionMenu(self, values=['---','and','or','not'], state='disabled')
        self.logic1.grid(row=2, column=3, padx=10, pady=10, sticky='w')

        self.tag2_label = customtkinter.CTkLabel(self, text="Tag2")
        self.tag2_label.grid(row=1, column=4, padx=10, pady=(10, 0), sticky='w')
        self.tag2 = customtkinter.CTkOptionMenu(self, values=self.task_df.Name.values, state='disabled')
        self.tag2.grid(row=2, column=4, padx=10, pady=10, sticky='w')

        self.add_tag_button = customtkinter.CTkButton(self, text='+', command=self.add_tag_button_group1_event, state='disabled')
        self.add_tag_button.grid(row=2, column=5, padx=10, pady=10, sticky='w')

        # Experiment Group2
        self.second_group_label = customtkinter.CTkLabel(self, text="2nd Effect")
        self.second_group_label.grid(row=3, column=1, padx=10, pady=(10, 0), sticky='n')

        self.tag1_group2 = customtkinter.CTkOptionMenu(self, values=self.task_df.Name.values, state='disabled')
        self.tag1_group2.grid(row=3, column=2, padx=10, pady=10, sticky='w')

        self.logic1_group2 = customtkinter.CTkOptionMenu(self, values=['---','and','or','not'], state='disabled')
        self.logic1_group2.grid(row=3, column=3, padx=10, pady=10, sticky='w')

        self.tag2_group2 = customtkinter.CTkOptionMenu(self, values=self.task_df.Name.values, state='disabled')
        self.tag2_group2.grid(row=3, column=4, padx=10, pady=10, sticky='w')

        self.add_tag_button_group2 = customtkinter.CTkButton(self, text='+', command=self.add_tag_button_group2_event, state='disabled')
        self.add_tag_button_group2.grid(row=3, column=5, padx=10, pady=10, sticky='w')

        # Apply button
        self.apply_button = customtkinter.CTkButton(self, text="Add Analysis", command=self.add_analysis)
        self.apply_button.grid(row=4, column=0, padx=10, pady=10, sticky='w')

        # References to dynamically created widgets
        self.logic2 = None
        self.tag3_label = None
        self.tag3 = None
        self.remove_tag_button = None

        self.logic2_group2 = None
        self.tag3_group2 = None
        self.remove_tag_button_group2 = None

    def set_controller(self, controller):
        self.controller = controller

    def tag_all_update_states(self, value):
        if value == 'all':
            self.logic1.configure(state='disabled')
            self.tag2.configure(state='disabled')
            self.add_tag_button.configure(state='disabled')
            if self.logic2 and self.tag3:
                self.logic2.configure(state='disabled')
                self.tag3.configure(state='disabled')
                self.remove_tag_button.configure(state='disabled')
        else:
            self.logic1.configure(state='normal')
            self.tag2.configure(state='normal')
            self.add_tag_button.configure(state='normal')
            if self.logic2 and self.tag3:
                self.logic2.configure(state='normal')
                self.tag3.configure(state='normal')
                self.remove_tag_button.configure(state='normal')


    def add_tag_button_group1_event(self):
        if self.tag3_label == None:
            self.tag3_label = customtkinter.CTkLabel(self, text="Tag3")
            self.tag3_label.grid(row=1, column=6, padx=10, pady=(10, 0), sticky='w')
        
        self.logic2 = customtkinter.CTkOptionMenu(self, values=['---','and','or','not'])
        self.logic2.grid(row=2, column=5, padx=10, pady=10, sticky='w')
        
        self.tag3 = customtkinter.CTkOptionMenu(self, values=self.task_df.Name.values)
        self.tag3.grid(row=2, column=6, padx=10, pady=10, sticky='w')

        self.remove_tag_button = customtkinter.CTkButton(self, text='-', command=self.remove_tag_button_group1_event)
        self.remove_tag_button.grid(row=2, column=7, padx=10, pady=10, sticky='w')

    def remove_tag_button_group1_event(self):
        self.logic2.destroy()
        self.logic2 = None
        if self.logic2_group2 == None:
            self.tag3_label.destroy()
            self.tag3_label = None
        self.tag3.destroy()
        self.tag3 = None
        self.remove_tag_button.destroy()
        self.remove_tag_button = None
        self.add_tag_button = customtkinter.CTkButton(self, text='+', command=self.add_tag_button_group1_event)
        self.add_tag_button.grid(row=2, column=5, padx=10, pady=10, sticky='w')


    def add_tag_button_group2_event(self):
        if self.tag3_label == None:
            self.tag3_label = customtkinter.CTkLabel(self, text="Tag3")
            self.tag3_label.grid(row=1, column=6, padx=10, pady=(10, 0), sticky='w')

        self.logic2_group2 = customtkinter.CTkOptionMenu(self, values=['---','and','or','not'])
        self.logic2_group2.grid(row=3, column=5, padx=10, pady=10, sticky='w')

        self.tag3_group2 = customtkinter.CTkOptionMenu(self, values=self.task_df.Name.values)
        self.tag3_group2.grid(row=3, column=6, padx=10, pady=10, sticky='w')

        self.remove_tag_button_group2 = customtkinter.CTkButton(self, text='-', command=self.remove_tag_button_group2_event)
        self.remove_tag_button_group2.grid(row=3, column=7, padx=10, pady=10, sticky='w')

    def remove_tag_button_group2_event(self):
        if self.logic2 == None:
            self.tag3_label.destroy()
            self.tag3_label = None
        self.logic2_group2.destroy()
        self.logic2_group2 = None
        self.tag3_group2.destroy()
        self.tag3_group2 = None
        self.remove_tag_button_group2.destroy()
        self.remove_tag_button_group2 = None
        self.add_tag_button_group2 = customtkinter.CTkButton(self, text='+', command=self.add_tag_button_group2_event)
        self.add_tag_button_group2.grid(row=3, column=5, padx=10, pady=10, sticky='w')

    def update_group2_state(self, value):
        if value in ["Standard Contrast", "Balanced Contrast"]:
            self.tag1_group2.configure(state='normal')
            self.logic1_group2.configure(state='normal')
            self.tag2_group2.configure(state='normal')
            self.add_tag_button_group2.configure(state='normal')
            if self.logic2_group2 and self.tag3_group2:
                self.logic2_group2.configure(state='normal')
                self.tag3_group2.configure(state='normal')
                self.remove_tag_button_group2(state='normal')
        else:
            self.tag1_group2.configure(state='disabled')
            self.logic1_group2.configure(state='disabled')
            self.tag2_group2.configure(state='disabled')
            self.add_tag_button_group2.configure(state='disabled')
            if self.logic2_group2 and self.tag3_group2:
                self.logic2_group2.configure(state='disabled')
                self.tag3_group2.configure(state='disabled')
                self.remove_tag_button_group2.configure(state='disabled')
    
    def add_analysis(self):
        return
    