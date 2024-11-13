import customtkinter


class AddAnalysisWindow(customtkinter.CTkToplevel):
    def __init__(self, master, controller):
        super().__init__(master)
        self.resizable(False, False)
        self.controller = controller
        self.task_df = self.controller.task_df
        self.title("Specify ALE Analyses")
        self.grid_columnconfigure(0, weight=1)
        self.create_widgets()

    def create_widgets(self):
        self.analysis_type_label = customtkinter.CTkLabel(self, text="Analysis Type")
        self.analysis_type_label.grid(
            row=1, column=0, padx=10, pady=(10, 0), sticky="w"
        )
        self.analysis_type = customtkinter.CTkOptionMenu(
            self,
            values=["Main Effect", "IPA", "Standard Contrast", "Balanced Contrast"],
            command=self.update_group2_state,
        )
        self.analysis_type.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.analysis_name_label = customtkinter.CTkLabel(self, text="Analysis Name")
        self.analysis_name_label.grid(
            row=1, column=1, padx=10, pady=(10, 0), sticky="w"
        )
        self.analysis_name = customtkinter.CTkEntry(self)
        self.analysis_name.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        # Experiment Group1
        self.tag1_label = customtkinter.CTkLabel(self, text="Tag 1")
        self.tag1_label.grid(row=1, column=2, padx=10, pady=(10, 0), sticky="w")
        self.tag1 = customtkinter.CTkOptionMenu(
            self,
            values=self.task_df.Name.values,
            command=lambda value: self.tag_enable_add_button(
                group=1, selected_value=value
            ),
        )
        self.tag1.grid(row=2, column=2, padx=10, pady=10, sticky="w")

        self.add_tag_button = customtkinter.CTkButton(
            self,
            text="+",
            state="disabled",
            command=lambda: self.add_tag_button_event(group=1),
        )
        self.add_tag_button.grid(row=2, column=3, padx=10, pady=10, sticky="w")

        self.tag_count_group1 = 1

        # Experiment Group2
        self.second_group_label = customtkinter.CTkLabel(self, text="2nd Effect")
        self.second_group_label.grid(row=3, column=1, padx=10, pady=(10, 0))
        self.tag1_group2 = customtkinter.CTkOptionMenu(
            self,
            values=self.task_df.Name.values[1:],
            state="disabled",
            command=lambda value: self.tag_enable_add_button(
                group=2, selected_value=value
            ),
        )
        self.tag1_group2.grid(row=3, column=2, padx=10, pady=10, sticky="w")

        self.add_tag_button_group2 = customtkinter.CTkButton(
            self,
            text="+",
            command=lambda: self.add_tag_button_event(group=2),
            state="disabled",
        )
        self.add_tag_button_group2.grid(row=3, column=3, padx=10, pady=10, sticky="w")

        self.tag_count_group2 = 1

        # Add analysis, import and reset buttons
        self.reset_entry_button = customtkinter.CTkButton(
            self, text="Reset Tags", command=self.reset_entry_button_event
        )
        self.reset_entry_button.grid(row=5, column=1, padx=10, pady=10, sticky="e")

        self.add_analysis_button = customtkinter.CTkButton(
            self,
            text="Add Analysis",
            command=self.add_analysis_button_event,
            fg_color="green4",
            hover_color="dark green",
        )
        self.add_analysis_button.grid(row=5, column=3, padx=10, pady=10, sticky="w")

        self.import_analysis_file_button = customtkinter.CTkButton(
            master=self,
            text="Import Analysis",
            command=self.import_analysis_file_button_event,
        )
        self.import_analysis_file_button.grid(row=5, column=0, padx=10, pady=10)

    def initial_state(self, group=[1, 2]):
        # Reset Experiment Group1
        if 1 in group:
            self.clear_group_widgets(1)
            self.tag1.grid(row=2, column=2, padx=10, pady=10, sticky="w")
            self.add_tag_button.grid(row=2, column=3, padx=10, pady=10, sticky="w")
            self.tag_count_group1 = 1

        # Reset Experiment Group2
        if 2 in group:
            self.clear_group_widgets(2)
            self.tag1_group2.grid(row=3, column=2, padx=10, pady=10, sticky="w")
            self.add_tag_button_group2.grid(
                row=3, column=3, padx=10, pady=10, sticky="w"
            )
            self.tag_count_group2 = 1

    def clear_group_widgets(self, group):
        if group == 1:
            widgets = self.grid_slaves(row=2)
            self.tag_count_group1 = 1
        else:
            widgets = self.grid_slaves(row=3)
            self.tag_count_group2 = 1

        for widget in widgets:
            if isinstance(widget, customtkinter.CTkOptionMenu) and widget not in [
                self.analysis_type,
                self.tag1,
                self.tag1_group2,
            ]:
                widget.destroy()

        # Clear tag labels in the first row
        for widget in self.grid_slaves(row=1):
            if widget.cget("text") not in ["Analysis Type", "Analysis Name", "Tag 1"]:
                widget.destroy()

    def tag_enable_add_button(self, group, selected_value):
        if selected_value != "all":
            if group == 1:
                self.add_tag_button.configure(state="normal")
            if group == 2:
                self.add_tag_button_group2.configure(state="normal")
        else:
            if group == 1:
                self.add_tag_button.configure(state="disabled")
                self.initial_state(group=[1])
            if group == 2:
                self.add_tag_button_group2.configure(state="disabled")
                self.initial_state(group=[2])

    def update_group2_state(self, value):
        state = (
            "normal"
            if value in ["Standard Contrast", "Balanced Contrast"]
            else "disabled"
        )

        widgets = self.grid_slaves(row=3)
        for widget in widgets:
            widget.configure(state=state)

    def add_tag_button_event(self, group):
        if group == 1:
            tag_count = self.tag_count_group1
            button = self.add_tag_button
        else:
            tag_count = self.tag_count_group2
            button = self.add_tag_button_group2

        # Determine current column of the button
        info = button.grid_info()
        current_column = info["column"]

        # Add logic and tag option menus
        logic_label = customtkinter.CTkLabel(self, text=f"Logic Operator {tag_count}")
        logic_label.grid(
            row=1, column=current_column, padx=10, pady=(10, 0), sticky="w"
        )
        logic_menu = customtkinter.CTkOptionMenu(self, values=["and", "or", "not"])
        logic_menu.grid(
            row=info["row"], column=current_column, padx=10, pady=10, sticky="w"
        )

        tag_label = customtkinter.CTkLabel(self, text=f"Tag {tag_count + 1}")
        tag_label.grid(
            row=1, column=current_column + 1, padx=10, pady=(10, 0), sticky="w"
        )
        tag_menu = customtkinter.CTkOptionMenu(
            self, values=self.task_df.Name.values[1:]
        )
        tag_menu.grid(
            row=info["row"], column=current_column + 1, padx=10, pady=10, sticky="w"
        )

        # Move button to the next column and limit to 3 logic fields and 4 tag fields
        if tag_count <= 2:
            button.grid(column=current_column + 2)

        if group == 1:
            self.tag_count_group1 += 1
        else:
            self.tag_count_group2 += 1

    def reset_entry_button_event(self):
        self.initial_state()

    def add_analysis_button_event(self):
        analysis_parameters = {
            "analysis_type": self.analysis_type.get(),
            "analysis_name": self.analysis_name.get(),
            "group1_logic": [self.tag1.get()],
        }

        # Collect group1 logic
        for i in range(1, self.tag_count_group1):
            logic = self.grid_slaves(row=2, column=3 + (i - 1) * 2)[0].get()
            tag = self.grid_slaves(row=2, column=4 + (i - 1) * 2)[0].get()
            analysis_parameters["group1_logic"].extend([logic, tag])

        if self.analysis_type.get() in ["Standard Contrast", "Balanced Contrast"]:
            analysis_parameters["group2_logic"] = [self.tag1_group2.get()]

            # Collect group2 logic
            for i in range(1, self.tag_count_group2):
                logic = self.grid_slaves(row=3, column=3 + (i - 1) * 2)[0].get()
                tag = self.grid_slaves(row=3, column=4 + (i - 1) * 2)[0].get()
                analysis_parameters["group2_logic"].extend([logic, tag])

        self.controller.get_analysis_parameters(analysis_parameters)
        # Add logic to handle the analysis_data dictionary

    def import_analysis_file_button_event(self):
        filename = customtkinter.filedialog.askopenfilename()
        if filename:
            self.controller.import_analysis_file(filename)
            print("Succesfully imported an analysis file.")
