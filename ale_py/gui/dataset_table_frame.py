import tkinter as tk
from tkinter import ttk

import customtkinter


class DatasetTableFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.lbl_title = customtkinter.CTkLabel(
            master=self,
            text="Dataset Info",
            justify=tk.LEFT,
            fg_color="#477AA2",
            corner_radius=5,
        )
        self.lbl_title.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 1))

        self.tree_frame = customtkinter.CTkFrame(self)
        self.tree_frame.grid(row=1, column=0, sticky="nsew")

        self.tree_frame.grid_rowconfigure(0, weight=1)
        self.tree_frame.grid_columnconfigure(0, weight=1)

        self.tree_scroll = ttk.Scrollbar(self.tree_frame)
        self.tree_scroll.grid(row=0, column=1, sticky="ns")

        # Define style for centered text
        style = ttk.Style()
        style.configure("Treeview.Heading", anchor=tk.CENTER)
        style.configure("Treeview", rowheight=25, font=("Arial", 12))
        # Remove borders
        style.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])

        self.tree = ttk.Treeview(
            self.tree_frame, yscrollcommand=self.tree_scroll.set, style="Treeview"
        )
        self.tree.grid(row=0, column=0, sticky="nsew")

        self.tree_scroll.config(command=self.tree.yview)

        self.tree["columns"] = ("Author", "Subjects", "Number of Foci")

        self.tree.column("#0", width=0, stretch=tk.NO)
        self.tree.column("Author", anchor=tk.CENTER, width=200)
        self.tree.column("Subjects", anchor=tk.CENTER, width=100)
        self.tree.column("Number of Foci", anchor=tk.CENTER, width=100)

        self.tree.heading("#0", text="", anchor=tk.CENTER)
        self.tree.heading("Author", text="Author", anchor=tk.CENTER)
        self.tree.heading("Subjects", text="Subjects", anchor=tk.CENTER)
        self.tree.heading("Number of Foci", text="Number of Foci", anchor=tk.CENTER)

    def set_controller(self, controller):
        self.controller = controller

    def fill_table(self, dataset_df):
        # Clear all existing entries in the Treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Insert new rows into the Treeview
        for row in dataset_df.itertuples(index=False):
            self.tree.insert(
                "", tk.END, values=(row.Articles, int(row.Subjects), row.NumberOfFoci)
            )
