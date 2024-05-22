import customtkinter
import tkinter

class ParameterWindow(customtkinter.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("ALE Parameters")
        self.grid_columnconfigure(0, weight=1)

        self.title_label = customtkinter.CTkLabel(self, text="Parameter Settings", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=10)

        # Cutoff prediction checkbox
        self.cutoff_prediction_var = tkinter.BooleanVar()
        self.cutoff_prediction_checkbox = customtkinter.CTkCheckBox(self, text="Cutoff Prediction", variable=self.cutoff_prediction_var)
        self.cutoff_prediction_checkbox.grid(row=1, column=0, padx=20, pady=10, sticky="w")

        # TFCE enabled checkbox
        self.tfce_enabled_var = tkinter.BooleanVar()
        self.tfce_enabled_checkbox = customtkinter.CTkCheckBox(self, text="TFCE Enabled", variable=self.tfce_enabled_var)
        self.tfce_enabled_checkbox.grid(row=2, column=0, padx=20, pady=10, sticky="w")

        # Monte-carlo iterations combobox (1000-10000)
        self.monte_carlo_iterations_label = customtkinter.CTkLabel(self, text="Monte-Carlo Iterations")
        self.monte_carlo_iterations_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.monte_carlo_combobox = customtkinter.CTkComboBox(self, values=["1000", "2500", "5000", "10000", "20000"])
        self.monte_carlo_combobox.set("5000")
        self.monte_carlo_combobox.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")

        # cFWE cluster-forming threshold combobox (0.00001, 0.0001, 0.001, 0.005, 0.01)
        self.cfwe_threshold_label = customtkinter.CTkLabel(self, text="cFWE Cluster-Forming Threshold")
        self.cfwe_threshold_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.cfwe_threshold_combobox = customtkinter.CTkComboBox(self, values=["0.00001", "0.0001", "0.001", "0.005", "0.01"])
        self.cfwe_threshold_combobox.set("0.001")
        self.cfwe_threshold_combobox.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Subsampling N combobox (500,1000,2500,5000,10000)
        self.subsampling_n_label = customtkinter.CTkLabel(self, text="IPA Subsampling N")
        self.subsampling_n_label.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
        self.subsampling_n_combobox = customtkinter.CTkComboBox(self, values=["500", "1000", "2500", "5000", "10000"])
        self.subsampling_n_combobox.set("2500")
        self.subsampling_n_combobox.grid(row=8, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Balanced contrast monte-carlo iterations combobox (500,1000,2000,3000,5000)
        self.balanced_contrast_label = customtkinter.CTkLabel(self, text="Balanced Contrast Monte-Carlo Iterations")
        self.balanced_contrast_label.grid(row=9, column=0, padx=20, pady=(10, 0), sticky="w")
        self.balanced_contrast_combobox = customtkinter.CTkComboBox(self, values=["500", "1000", "2000", "3000", "5000"])
        self.balanced_contrast_combobox.set("1000")
        self.balanced_contrast_combobox.grid(row=10, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Apply button
        self.apply_button = customtkinter.CTkButton(self, text="Apply", command=self.apply_parameters)
        self.apply_button.grid(row=11, column=0, padx=20, pady=10)

    def apply_parameters(self):
        # Here you would implement the logic to apply the parameters to your calculations.
        parameters = {
            "cutoff_prediction": self.cutoff_prediction_var.get(),
            "tfce_enabled": self.tfce_enabled_var.get(),
            "monte_carlo_iterations": self.monte_carlo_combobox.get(),
            "cfwe_threshold": float(self.cfwe_threshold_combobox.get()),
            "subsampling_n": int(self.subsampling_n_combobox.get()),
            "balanced_contrast_iterations": int(self.balanced_contrast_combobox.get())
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