import customtkinter
import tkinter
from core.utils.input import load_excel

class AnalysisTableFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure((1,2), weight=1)

        self.lbl_title = customtkinter.CTkLabel(
            master=self,
            text=f"Analysis Info",
            justify=tkinter.LEFT,
        )
        self.lbl_title.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(5,0))
        
        label = customtkinter.CTkLabel(self, text=f"Analysis Type", fg_color="#477AA2", corner_radius=4)
        label.grid(row=1, column=0, sticky='nsew', padx=(10,2), pady=(10,0))
        
        label = customtkinter.CTkLabel(self, text=f"Analysis Name", fg_color="#477AA2", corner_radius=4)
        label.grid(row=1, column=1, sticky='nsew', padx=(0,2), pady=(10,0))

        label = customtkinter.CTkLabel(self, text=f"Analysis Tags", fg_color="#477AA2", corner_radius=4)
        label.grid(row=1, column=2, sticky='nsew', padx=(0,10), pady=(10,0))

        self.scroll_table = customtkinter.CTkScrollableFrame(self)
        self.scroll_table.grid(row=2, column=0, columnspan=3, sticky='nsew', padx=10, pady=10)
        
        self.scroll_table.grid_columnconfigure((1,2), weight=1)
    
    def set_controller(self, controller):
        self.controller = controller

    def fill_table(self, analysis_df):
        for row in range(analysis_df.shape[0]):
            row_content = analysis_df.iloc[row]

            analysis_type = row_content.iloc[0]
            label = customtkinter.CTkLabel(self.scroll_table, text=f"{analysis_type:^12}")
            label.grid(row=row, column=0, sticky='nsew', padx=(10,2), pady=10)

            analysis_name = row_content[1]
            label = customtkinter.CTkLabel(self.scroll_table, text=f"{analysis_name:^70}")
            label.grid(row=row, column=1, sticky='nsew', padx=(0,2), pady=10)

            analysis_conditions = row_content[2:].dropna().str.lower().str.strip().values
            label = customtkinter.CTkLabel(self.scroll_table, text=f"{analysis_conditions}")
            label.grid(row=row, column=2, sticky='nsew', padx=(0,10), pady=10)