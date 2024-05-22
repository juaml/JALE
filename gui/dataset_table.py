import customtkinter
import tkinter
from core.utils.input import read_experiment_info

class DatasetTable(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure((0,1,2),weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.lbl_title = customtkinter.CTkLabel(
            master=self,
            text=f"Dataset Info",
            justify=tkinter.LEFT,
        )
        self.lbl_title.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(5,0))
        
        label = customtkinter.CTkLabel(self, text=f"Author".center(60), fg_color="#477AA2", corner_radius=4)
        label.grid(row=1, column=0, sticky='nsew', padx=(10,2), pady=(10,0))
        
        label = customtkinter.CTkLabel(self, text=f"Subjects".center(25), fg_color="#477AA2", corner_radius=4)
        label.grid(row=1, column=1, sticky='nsew', padx=(0,2), pady=(10,0))

        label = customtkinter.CTkLabel(self, text=f"Number of Foci".center(25), fg_color="#477AA2", corner_radius=4)
        label.grid(row=1, column=2, sticky='nsew', padx=(0,10), pady=(10,0))

        self.scroll_table = customtkinter.CTkScrollableFrame(self)
        self.scroll_table.grid(row=2, column=0, columnspan=3, sticky='nsew', padx=10, pady=10)
        
        self.scroll_table.grid_columnconfigure((0,1,2), weight=1)

    def fill_table(self, dataset_file):
        exp_info, _ = read_experiment_info(dataset_file)

        for row in range(exp_info.shape[0]):
            row_content = exp_info.iloc[row]

            article = row_content.Articles
            label = customtkinter.CTkLabel(self.scroll_table, text=f"{article:^35}")
            label.grid(row=row, column=0, sticky='nsew', padx=(10,2), pady=10)

            subjects = int(row_content.Subjects)
            label = customtkinter.CTkLabel(self.scroll_table, text=f"{subjects:^25}")
            label.grid(row=row, column=1, sticky='nsew', padx=(10,2), pady=10)

            number_of_foci = row_content.NumberOfFoci
            label = customtkinter.CTkLabel(self.scroll_table, text=f"{number_of_foci:^25}")
            label.grid(row=row, column=2, sticky='nsew', padx=(10,2), pady=10)

