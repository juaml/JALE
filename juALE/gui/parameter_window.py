import tkinter
import tkinter.messagebox as messagebox

import customtkinter


class ParameterWindow(customtkinter.CTkToplevel):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        self.title("ALE Parameters")
        self.grid_columnconfigure(0, weight=1)

        self.title_label = customtkinter.CTkLabel(
            self,
            text="Parameter Settings",
            font=customtkinter.CTkFont(size=20, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=10)

        # Cutoff prediction checkbox
        self.cutoff_prediction_var = tkinter.BooleanVar(value=True)
        self.cutoff_prediction_checkbox = customtkinter.CTkCheckBox(
            self, text="Cutoff Prediction", variable=self.cutoff_prediction_var
        )
        self.cutoff_prediction_checkbox.grid(
            row=1, column=0, padx=20, pady=10, sticky="w"
        )

        # TFCE enabled checkbox
        self.tfce_enabled_var = tkinter.BooleanVar(value=True)
        self.tfce_enabled_checkbox = customtkinter.CTkCheckBox(
            self, text="TFCE Enabled", variable=self.tfce_enabled_var
        )
        self.tfce_enabled_checkbox.grid(row=2, column=0, padx=20, pady=10, sticky="w")

        # Monte-carlo iterations textbox
        self.monte_carlo_iterations_label = customtkinter.CTkLabel(
            self, text="Monte-Carlo Iterations"
        )
        self.monte_carlo_iterations_label.grid(
            row=3, column=0, padx=20, pady=(10, 0), sticky="w"
        )
        self.monte_carlo_iterations_textbox = customtkinter.CTkEntry(self)
        self.monte_carlo_iterations_textbox.insert(0, "5000")
        self.monte_carlo_iterations_textbox.grid(
            row=4, column=0, padx=20, pady=(0, 10), sticky="ew"
        )

        # cFWE cluster-forming threshold textbox (0.00001, 0.0001, 0.001, 0.005, 0.01)
        self.cfwe_threshold_label = customtkinter.CTkLabel(
            self, text="cFWE Cluster-Forming Threshold"
        )
        self.cfwe_threshold_label.grid(
            row=5, column=0, padx=20, pady=(10, 0), sticky="w"
        )
        self.cfwe_threshold_textbox = customtkinter.CTkEntry(self)
        self.cfwe_threshold_textbox.insert(0, "0.001")
        self.cfwe_threshold_textbox.grid(
            row=6, column=0, padx=20, pady=(0, 10), sticky="ew"
        )

        # Subsampling N textbox (500,1000,2500,5000,10000)
        self.subsampling_n_label = customtkinter.CTkLabel(
            self, text="IPA Subsampling N"
        )
        self.subsampling_n_label.grid(
            row=7, column=0, padx=20, pady=(10, 0), sticky="w"
        )
        self.subsampling_n_textbox = customtkinter.CTkEntry(self)
        self.subsampling_n_textbox.insert(0, "2500")
        self.subsampling_n_textbox.grid(
            row=8, column=0, padx=20, pady=(0, 10), sticky="ew"
        )

        # Balanced contrast monte-carlo iterations textbox (500,1000,2000,3000,5000)
        self.balanced_contrast_label = customtkinter.CTkLabel(
            self, text="Balanced Contrast Monte-Carlo Iterations"
        )
        self.balanced_contrast_label.grid(
            row=9, column=0, padx=20, pady=(10, 0), sticky="w"
        )
        self.balanced_contrast_textbox = customtkinter.CTkEntry(self)
        self.balanced_contrast_textbox.insert(0, "1000")
        self.balanced_contrast_textbox.grid(
            row=10, column=0, padx=20, pady=(0, 10), sticky="ew"
        )

        # Apply button
        self.apply_button = customtkinter.CTkButton(
            self, text="Apply", command=self.apply_parameters
        )
        self.apply_button.grid(row=11, column=0, padx=20, pady=10)

    def validate_monte_carlo_iterations(self):
        valid = True
        try:
            value = int(self.monte_carlo_iterations_textbox.get())
            if value < 100 or value > 100000:
                messagebox.showerror(
                    "Invalid Input",
                    "Monte Carlo iterations must be a whole number between 100 and 100,000.",
                )
                self.monte_carlo_iterations_textbox.delete(0, "end")
                self.monte_carlo_iterations_textbox.insert(0, "5000")
                valid = False
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Monte Carlo iterations must be a whole number between 100 and 100,000.",
            )
            self.monte_carlo_iterations_textbox.delete(0, "end")
            self.monte_carlo_iterations_textbox.insert(0, "5000")
            valid = False
        return valid

    def validate_cluster_forming_threshold(self):
        valid = True
        try:
            value = float(self.cfwe_threshold_textbox.get())
            if value < 0.00001 or value > 0.05:
                messagebox.showerror(
                    "Invalid Input",
                    "cFWE cluster forming threshold must be between 0.00001 and 0.05.",
                )
                self.cfwe_threshold_textbox.delete(0, "end")
                self.cfwe_threshold_textbox.insert(0, "0.001")
                valid = False
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "cFWE cluster forming threshold must be between 0.00001 and 0.05.",
            )
            self.cfwe_threshold_textbox.delete(0, "end")
            self.cfwe_threshold_textbox.insert(0, "0.001")
            valid = False
        return valid

    def validate_subsampling_n(self):
        try:
            value = int(self.subsampling_n_textbox.get())
            if value < 100 or value > 25000:
                messagebox.showerror(
                    "Invalid Input",
                    "IPA subsampling N must be a whole number between 100 and 25000.",
                )
                self.subsampling_n_textbox.delete(0, "end")
                self.subsampling_n_textbox.insert(0, "2500")
                return False
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "IPA subsampling N must be a whole number between 100 and 25000.",
            )
            self.subsampling_n_textbox.delete(0, "end")
            self.subsampling_n_textbox.insert(0, "2500")
            return False

    def validate_balanced_contrast_iterations(self):
        valid = True
        try:
            value = int(self.balanced_contrast_textbox.get())
            if value < 100 or value > 10000:
                messagebox.showerror(
                    "Invalid Input",
                    "Balanced contrast monte-carlo iterations must be a whole number between 100 and 10000.",
                )
                self.balanced_contrast_textbox.delete(0, "end")
                self.balanced_contrast_textbox.insert(0, "1000")
                valid = False
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Balanced contrast monte-carlo iterations must be a whole number between 100 and 10000.",
            )
            self.balanced_contrast_textbox.delete(0, "end")
            self.balanced_contrast_textbox.insert(0, "1000")
            valid = False
        return valid

    def apply_parameters(self):
        val_mc = self.validate_monte_carlo_iterations()
        val_cft = self.validate_cluster_forming_threshold()
        val_subn = self.validate_subsampling_n()
        val_bci = self.validate_balanced_contrast_iterations()

        if any([val_mc, val_cft, val_subn, val_bci]) is False:
            return

        parameters = {
            "cutoff_prediction": self.cutoff_prediction_var.get(),
            "tfce_enabled": self.tfce_enabled_var.get(),
            "monte_carlo_iterations": int(self.monte_carlo_iterations_textbox.get()),
            "cfwe_threshold": float(self.cfwe_threshold_textbox.get()),
            "subsampling_n": int(self.subsampling_n_textbox.get()),
            "balanced_contrast_iterations": int(self.balanced_contrast_textbox.get()),
        }
        self.controller.get_ale_parameters(parameters)

        self.apply_button.configure(fg_color="green")
        self.apply_button.configure(text="Applied")
        print("Changed ALE Parameters")
        self.after(
            2000, lambda: self.apply_button.configure(fg_color="#20548c", text="Apply")
        )


class ParameterWarningWindow(customtkinter.CTkToplevel):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller
        self.title("WARNING!")
        self.warning_label = customtkinter.CTkLabel(
            self,
            text="Default parameter values are set automatically.\nChanging parameters only advised for experts.",
            font=customtkinter.CTkFont(size=16),
        )
        self.warning_label.grid(row=0, column=0, columnspan=2, padx=20, pady=10)

        self.understand_button = customtkinter.CTkButton(
            self, text="I understand.", command=self.understand_button_event
        )
        self.understand_button.grid(row=1, column=0, padx=20, pady=10)

        self.close_button = customtkinter.CTkButton(
            self, text="Close", command=self.close_button_event
        )
        self.close_button.grid(row=1, column=1, padx=20, pady=10)

    def understand_button_event(self):
        self.controller.open_parameter_window()
        self.destroy()

    def close_button_event(self):
        self.destroy()
