import customtkinter
import tkinter
import tkinter.messagebox as messagebox

class ParameterWindow(customtkinter.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("ALE Parameters")
        self.grid_columnconfigure(0, weight=1)

        self.title_label = customtkinter.CTkLabel(self, text="Parameter Settings", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=10)

        # Cutoff prediction checkbox
        self.cutoff_prediction_var = tkinter.BooleanVar(value=True)
        self.cutoff_prediction_checkbox = customtkinter.CTkCheckBox(self, text="Cutoff Prediction", variable=self.cutoff_prediction_var)
        self.cutoff_prediction_checkbox.grid(row=1, column=0, padx=20, pady=10, sticky="w")

        # TFCE enabled checkbox
        self.tfce_enabled_var = tkinter.BooleanVar(value=True)
        self.tfce_enabled_checkbox = customtkinter.CTkCheckBox(self, text="TFCE Enabled", variable=self.tfce_enabled_var)
        self.tfce_enabled_checkbox.grid(row=2, column=0, padx=20, pady=10, sticky="w")
        # Monte-carlo iterations textbox
        self.monte_carlo_iterations_label = customtkinter.CTkLabel(self, text="Monte-Carlo Iterations")
        self.monte_carlo_iterations_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.monte_carlo_iterations_textbox = customtkinter.CTkEntry(self, validate='focusout', validatecommand=self.validate_monte_carlo_iterations)
        self.monte_carlo_iterations_textbox.insert(0, "5000")
        self.monte_carlo_iterations_textbox.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Validate Monte Carlo iterations input
        self.monte_carlo_iterations_textbox.bind("<FocusOut>", self.validate_monte_carlo_iterations())

        # cFWE cluster-forming threshold textbox (0.00001, 0.0001, 0.001, 0.005, 0.01)
        self.cfwe_threshold_label = customtkinter.CTkLabel(self, text="cFWE Cluster-Forming Threshold")
        self.cfwe_threshold_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.cfwe_threshold_textbox = customtkinter.CTkEntry(self)
        self.cfwe_threshold_textbox.insert(0, "0.001")
        self.cfwe_threshold_textbox.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Subsampling N textbox (500,1000,2500,5000,10000)
        self.subsampling_n_label = customtkinter.CTkLabel(self, text="IPA Subsampling N")
        self.subsampling_n_label.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
        self.subsampling_n_textbox = customtkinter.CTkEntry(self)
        self.subsampling_n_textbox.insert(0, "2500")
        self.subsampling_n_textbox.grid(row=8, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Balanced contrast monte-carlo iterations textbox (500,1000,2000,3000,5000)
        self.balanced_contrast_label = customtkinter.CTkLabel(self, text="Balanced Contrast Monte-Carlo Iterations")
        self.balanced_contrast_label.grid(row=9, column=0, padx=20, pady=(10, 0), sticky="w")
        self.balanced_contrast_textbox = customtkinter.CTkEntry(self)
        self.balanced_contrast_textbox.insert(0, "1000")
        self.balanced_contrast_textbox.grid(row=10, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Apply button
        self.apply_button = customtkinter.CTkButton(self, text="Apply", command=self.apply_parameters)
        self.apply_button.grid(row=11, column=0, padx=20, pady=10)

    def validate_monte_carlo_iterations(self):
        try:
            value = int(self.monte_carlo_iterations_textbox.get())
            if value < 100 or value > 100000:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Monte Carlo iterations must be a whole number between 100 and 100,000.")
            self.monte_carlo_iterations_textbox.delete(0, "end")
            self.monte_carlo_iterations_textbox.insert(0, "5000")

    def apply_parameters(self):
    # Here you would implement the logic to apply the parameters to your calculations.
        parameters = {
            "cutoff_prediction": self.cutoff_prediction_var.get(),
            "tfce_enabled": self.tfce_enabled_var.get(),
            "monte_carlo_iterations": int(self.monte_carlo_iterations_textbox.get()),
            "cfwe_threshold": float(self.cfwe_threshold_textbox.get()),
            "subsampling_n": int(self.subsampling_n_textbox.get()),
            "balanced_contrast_iterations": int(self.balanced_contrast_textbox.get())
        }
        print("Applied parameters:", parameters)
        # Add your logic to pass these parameters to your calculations


        '''
        Cutoff prediction checkbox
        TFCE enabled checkbox
        Monte-carlo iterations slider (1000-10000)
        cFWE cluster-forming threshold slider (0.00001, 0.0001, 0.001, 0.005, 0.01)
        Subsampling N slider (500,1000,2500,5000,10000)
        Balanced contrast monte-carlo iterations slider (500,1000,2000,3000,5000)
        '''