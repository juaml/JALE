import logging
from threading import Thread

import customtkinter

from jale.ale import (
    load_config,
    load_dataframes,
    run_balanced_contrast,
    run_contrast_analysis,
    run_ma_clustering,
    run_main_effect,
    run_probabilistic_ale,
    setup_logger,
    setup_project_folder,
)


class Controller:
    def __init__(
        self, sidebar_frame, analysis_table_frame, dataset_table_frame, output_log_frame
    ):
        # Frames
        self.sidebar_frame = sidebar_frame
        self.analysis_table_frame = analysis_table_frame
        self.dataset_table_frame = dataset_table_frame
        self.output_log_frame = output_log_frame

        # ALE objects
        self.analysis_df = None
        self.dataset_df = None
        self.task_df = None
        self.params = None
        self.clustering_params = None
        self.config = None
        self.project_path = None

    # Sidebar Buttons
    def config_to_gui(self):
        filename = customtkinter.filedialog.askopenfilename()
        if filename:
            self.config = load_config(filename)
            self.project_path = setup_project_folder(self.config)

            logger = setup_logger(self.project_path)
            self.output_log_frame.set_logger(logger)
            logger.info("Logger initialized and project setup complete.")
            self.params = self.config.get("parameters", {})
            self.clustering_params = self.config.get("clustering_params", {})
            self.dataset_df, self.task_df, self.analysis_df = load_dataframes(
                self.project_path, self.config
            )
            self.dataset_table_frame.fill_dataset_table(self.dataset_df)
            analysis_df_formatted = (
                self.analysis_table_frame.format_imported_analysis_file(
                    self.analysis_df
                )
            )
            self.analysis_table_frame.fill_analysis_table(analysis_df_formatted)

    def run_analysis(self):
        logger = logging.getLogger("ale_logger")

        def process_analyses():
            """
            Run analyses sequentially in threads.
            """
            for row_idx in range(self.analysis_df.shape[0]):
                # Skip empty rows
                if not isinstance(self.analysis_df.iloc[row_idx, 0], str):
                    continue

                thread = None

                if self.analysis_df.iloc[row_idx, 0] == "M":
                    thread = Thread(
                        target=run_main_effect,
                        args=(
                            self.analysis_df,
                            row_idx,
                            self.project_path,
                            self.params,
                            self.dataset_df,
                            self.task_df,
                        ),
                    )

                elif self.analysis_df.iloc[row_idx, 0][0] == "P":
                    thread = Thread(
                        target=run_probabilistic_ale,
                        args=(
                            self.analysis_df,
                            row_idx,
                            self.project_path,
                            self.params,
                            self.dataset_df,
                            self.task_df,
                        ),
                    )

                elif self.analysis_df.iloc[row_idx, 0] == "C":
                    thread = Thread(
                        target=run_contrast_analysis,
                        args=(
                            self.analysis_df,
                            row_idx,
                            self.project_path,
                            self.params,
                            self.dataset_df,
                            self.task_df,
                        ),
                    )

                elif self.analysis_df.iloc[row_idx, 0][0] == "B":
                    thread = Thread(
                        target=run_balanced_contrast,
                        args=(
                            self.analysis_df,
                            row_idx,
                            self.project_path,
                            self.params,
                            self.dataset_df,
                            self.task_df,
                        ),
                    )

                elif self.analysis_df.iloc[row_idx, 0] == "Cluster":
                    thread = Thread(
                        target=run_ma_clustering,
                        args=(
                            self.analysis_df,
                            row_idx,
                            self.project_path,
                            self.clustering_params,
                            self.dataset_df,
                            self.task_df,
                        ),
                    )

                if thread:
                    thread.start()
                    thread.join()  # Wait for the current thread to finish

            logger.info("Analysis completed.")

        # Run analyses in a background thread to keep the GUI responsive
        background_thread = Thread(target=process_analyses)
        background_thread.start()

    def stop_analysis(self):
        return
