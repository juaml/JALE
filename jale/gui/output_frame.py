import sys
import tkinter

import customtkinter

# CustomText widget for displaying the output


class CustomText(tkinter.Text):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(state=tkinter.DISABLED)
        # custom_font = tkinter.font.Font(family="Helvetica", size=12)
        # self.configure(font=custom_font)

    def write(self, message):
        self.configure(state=tkinter.NORMAL)
        self.insert(tkinter.END, message)
        self.configure(state=tkinter.DISABLED)
        self.see(tkinter.END)

    def flush(self):
        pass


# Redirect standard output to the CustomText widget


class RedirectedStdout:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.write(message)

    def flush(self):
        pass


# Frame to hold the CustomText widget


class OutputFrame(customtkinter.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.text_widget = CustomText(self)
        self.text_widget.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = customtkinter.CTkScrollbar(
            self, command=self.text_widget.yview
        )
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.text_widget["yscrollcommand"] = self.scrollbar.set

        self.redirect_stdout()

    def redirect_stdout(self):
        sys.stdout = RedirectedStdout(self.text_widget)
