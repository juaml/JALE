import customtkinter

class AleParameterWindow(customtkinter.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.geometry("400x300")

        self.label = customtkinter.CTkLabel(self, text="Ale Parameters")
        self.label.pack(padx=20, pady=20)