import customtkinter
import tkinter

class AnalysisInfoTable(customtkinter.CTkFrame):
    def __init__(self, master, rows, columns):
        super().__init__(master)
        self.grid_rowconfigure(tuple(range(rows)), weight=0)
        self.grid_columnconfigure(2, weight=1)

        self.lbl_title = customtkinter.CTkLabel(
            master=self,
            text=f"Analysis Info",
            justify=tkinter.LEFT,
        )
        self.lbl_title.grid(row=0, column=0, sticky="wns", padx=10, pady=(5,0))
        
        label = customtkinter.CTkLabel(self, text=f"Analysis Type", fg_color="#477AA2", corner_radius=4)
        label.grid(row=1, column=0, sticky='nsew', padx=(10,2), pady=(10,0))
        
        label = customtkinter.CTkLabel(self, text=f"Analysis Name", fg_color="#477AA2", corner_radius=4)
        label.grid(row=1, column=1, sticky='nsew', padx=(0,2), pady=(10,0))

        label = customtkinter.CTkLabel(self, text=f"Analysis Tags", fg_color="#477AA2", corner_radius=4)
        label.grid(row=1, column=2, sticky='nsew', padx=(0,10), pady=(10,0))

    def fill_table(self):
        for row in range(self.rows):
            for column in range(self.columns):
                label = customtkinter.CTkLabel(self, text=f"{row, column}")
                label.grid(row=row+2, column=column, sticky='nsew', padx=10, pady=10)
