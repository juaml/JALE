import customtkinter
import numpy as np
import customtkinter
import tkinter as tk
from tkinter import ttk
import pandas as pd

class AnalysisTableFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.analysis_type_abbreviation_dict = {'Main Effect': 'M', 'IPA': 'P', 'Standard Contrast': 'C', 'Balanced Contrast': 'B'}
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.lbl_title = customtkinter.CTkLabel(
            master=self,
            text="Analysis Info",
            justify=tk.LEFT,
            fg_color="#477AA2",
            corner_radius=5
        )
        self.lbl_title.grid(row=0, column=0, sticky="ew", pady=(0,1))

        self.tree_frame = customtkinter.CTkFrame(self)
        self.tree_frame.grid(row=1, column=0, sticky="nsew")

        self.tree_frame.grid_rowconfigure(0, weight=1)
        self.tree_frame.grid_columnconfigure(0, weight=1)

        self.tree_scroll = ttk.Scrollbar(self.tree_frame)
        self.tree_scroll.grid(row=0, column=1, sticky="ns")

        self.tree = ttk.Treeview(self.tree_frame, yscrollcommand=self.tree_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")

        self.tree_scroll.config(command=self.tree.yview)

        self.tree['columns'] = ("Analysis Type", "Analysis Name", "Group 1 Logic", "Group 2 Logic")

        self.tree.column("#0", width=0, stretch=tk.NO)
        self.tree.column("Analysis Type", anchor=tk.CENTER, width=100,  stretch=tk.NO)
        self.tree.column("Analysis Name", anchor=tk.CENTER, width=250,  stretch=tk.NO)
        self.tree.column("Group 1 Logic", anchor=tk.CENTER, width=280,  stretch=tk.NO)
        self.tree.column("Group 2 Logic", anchor=tk.CENTER, width=280,  stretch=tk.NO)

        self.tree.heading("#0", text="", anchor=tk.CENTER)
        self.tree.heading("Analysis Type", text="Analysis Type", anchor=tk.CENTER)
        self.tree.heading("Analysis Name", text="Analysis Name", anchor=tk.CENTER)
        self.tree.heading("Group 1 Logic", text="Group 1 Logic", anchor=tk.CENTER)
        self.tree.heading("Group 2 Logic", text="Group 2 Logic", anchor=tk.CENTER)

    def set_controller(self, controller):
        self.controller = controller

    def reset_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def fill_table(self, analysis_df):
        # Clear all existing entries in the Treeview
        self.reset_table()
        # Insert new rows into the Treeview
        for row in analysis_df.itertuples(index=False):
            analysis_type = self.analysis_type_abbreviation_dict[row[0]]
            self.tree.insert("", tk.END, values=(analysis_type, row[1], row[2], row[3]))

    def format_imported_analysis_file(self, analysis_df):
        self.reset_table()
        df_format = pd.DataFrame(columns=['analysis_type', 'analysis_name', 'group1_logic', 'group2_logic'])
        skip_row = False
        for row in range(analysis_df.shape[0]):
            if skip_row == True:
                skip_row = False
                continue   

            analysis_type = analysis_df.iloc[row,0]
            if analysis_type in ['B','C']:
                analysis_name = f"{analysis_df.iloc[row,1]} vs. {analysis_df.iloc[row+1,0]}"
                group1_logic = analysis_df.iloc[row,2:].dropna().str.lower().str.strip().values
                group2_logic = analysis_df.iloc[row+1,2:].dropna().str.lower().str.strip().values
                new_row = pd.DataFrame([analysis_type, analysis_name, group1_logic, group2_logic])
                skip_row = True
            else:
                analysis_name = analysis_df.iloc[row,1]
                group1_logic = analysis_df.iloc[row,2:].dropna().str.lower().str.strip().values
                self.tree.insert("", tk.END, values=(analysis_type, analysis_name, group1_logic, '----'))