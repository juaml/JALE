import logging
import tkinter

import customtkinter


class TextWidgetHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        """
        Emit a log record to the text widget.
        """
        log_entry = self.format(record)
        self.text_widget.update_log(log_entry)


class OutputLogFrame(customtkinter.CTkFrame):
    def __init__(self, master, name):
        """
        Creates a 2x1 frame with a text box for outputting log messages.
        """
        super().__init__(master)
        self.name = name

        # configure grid layout (1x1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)  # log row will resize
        self.columnconfigure(0, weight=1)

        # Frame title
        self.lbl_title = customtkinter.CTkLabel(
            master=self,
            text=f"{self.name}",
            justify=tkinter.LEFT,
        )
        self.lbl_title.grid(row=0, column=0, sticky="wns", padx=15, pady=15)

        # Text Box
        self.txt_log = tkinter.Text(
            master=self,
            wrap=tkinter.WORD,
            bg="#343638",
            fg="#ffffff",
            padx=20,
            pady=5,
            spacing1=4,  # spacing before a line
            spacing3=4,  # spacing after a line / wrapped line
            cursor="arrow",
            font=("Courier", 14),
        )
        self.txt_log.grid(row=1, column=0, padx=(15, 0), pady=(0, 15), sticky="nsew")
        self.txt_log.configure(state=tkinter.DISABLED)

        # Scrollbar
        self.scrollbar = customtkinter.CTkScrollbar(
            master=self, command=self.txt_log.yview
        )
        self.scrollbar.grid(row=1, column=1, padx=(0, 15), pady=(0, 15), sticky="ns")

        # Connect textbox scroll event to scrollbar
        self.txt_log.configure(yscrollcommand=self.scrollbar.set)

        self.controller = None

    def set_controller(self, controller):
        self.controller = controller

    def update_log(self, msg, overwrite=False):
        """
        Called from controller. Updates the log with given message. If overwrite is True,
        the last line will be cleared before the message is added.
        """
        self.txt_log.configure(state=tkinter.NORMAL)
        if overwrite:
            self.txt_log.delete("end-1c linestart", "end")
        self.txt_log.insert(tkinter.END, "\n" + msg)
        self.txt_log.configure(state=tkinter.DISABLED)
        self.txt_log.see(tkinter.END)

    def clear_log(self):
        """
        Called from controller. Clears the log.
        """
        self.txt_log.configure(state=tkinter.NORMAL)
        self.txt_log.delete(1.0, tkinter.END)
        self.txt_log.configure(state=tkinter.DISABLED)
        self.txt_log.see(tkinter.END)

    def set_logger(self, logger):
        """
        Attaches the logger to this frame's text widget.
        """
        self.logger = logger
        handler = TextWidgetHandler(self)
        handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        if len(logger.handlers) < 3:
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)  # Set the desired log level
