"""
RFID Processing Worker Thread.

Handles background processing of RFID pipeline to prevent UI freezing.
"""

from PyQt5.QtCore import QThread, pyqtSignal
import traceback

from .config import RFIDConfig
from .core import (
    RFIDPreprocessor,
    BoutDetector,
    GBIGenerator,
    SocialNetworkAnalyzer,
    EdgelistGenerator,
    DisplacementDetector,
    HindeIndexCalculator
)


class RFIDProcessingWorker(QThread):
    """
    Worker thread for RFID data processing.

    Signals:
        progress_updated(int, str): Emits (percentage, status_message)
        step_completed(str, str): Emits (step_name, output_file_path)
        processing_complete(bool, str): Emits (success, message)
        error_occurred(str): Emits error message
    """

    progress_updated = pyqtSignal(int, str)
    step_completed = pyqtSignal(str, str)
    processing_complete = pyqtSignal(bool, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, config: RFIDConfig, enabled_steps: dict):
        """
        Initialize worker thread.

        Args:
            config: RFID configuration object
            enabled_steps: Dictionary of step_name: enabled (bool)
        """
        super().__init__()
        self.config = config
        self.enabled_steps = enabled_steps

        # Storage for intermediate results
        self.rfid_df = None
        self.movebout_df = None
        self.gbi_dict = None
        self.metadata_df = None

    def run(self):
        """
        Execute the RFID processing pipeline.

        Runs enabled pipeline steps sequentially.
        """
        try:
            total_steps = sum(1 for enabled in self.enabled_steps.values() if enabled)
            current_step = 0

            # Step 1: Raw RFID Processing
            if self.enabled_steps.get('raw_rfid', True):
                current_step += 1
                self.progress_updated.emit(
                    int(current_step / total_steps * 100),
                    f"Step {current_step}/{total_steps}: Processing raw RFID data..."
                )

                preprocessor = RFIDPreprocessor(self.config)
                self.rfid_df = preprocessor.process_raw_rfid(
                    progress_callback=lambda msg: self.progress_updated.emit(
                        int(current_step / total_steps * 100), msg
                    )
                )

                # Store metadata for later use
                self.metadata_df = preprocessor.metadata_df

                self.step_completed.emit("Raw RFID Processing", "ALLTRIAL_RFID_DATA.csv")

            # Step 2: Movement Bout Detection
            if self.enabled_steps.get('bout_detection', True):
                current_step += 1
                self.progress_updated.emit(
                    int(current_step / total_steps * 100),
                    f"Step {current_step}/{total_steps}: Detecting movement bouts..."
                )

                if self.rfid_df is None:
                    raise ValueError("Raw RFID data not available. Enable 'Raw RFID Processing' first.")

                bout_detector = BoutDetector(self.config)
                self.movebout_df = bout_detector.detect_bouts(
                    self.rfid_df,
                    progress_callback=lambda msg: self.progress_updated.emit(
                        int(current_step / total_steps * 100), msg
                    )
                )

                self.step_completed.emit("Movement Bout Detection", "ALLTRIAL_MOVEBOUT.csv")

            # Step 3: GBI Matrix Generation
            if self.enabled_steps.get('gbi', True):
                current_step += 1
                self.progress_updated.emit(
                    int(current_step / total_steps * 100),
                    f"Step {current_step}/{total_steps}: Creating GBI matrices..."
                )

                if self.movebout_df is None:
                    raise ValueError("Movement bout data not available. Enable 'Movement Bout Detection' first.")

                gbi_generator = GBIGenerator(self.config)
                self.gbi_dict = gbi_generator.create_gbi_matrices(
                    self.movebout_df,
                    self.metadata_df,
                    progress_callback=lambda msg: self.progress_updated.emit(
                        int(current_step / total_steps * 100), msg
                    )
                )

                self.step_completed.emit("GBI Matrix Generation", "*_MOVEBOUT_GBI.csv")

            # Step 4: Social Network Analysis
            if self.enabled_steps.get('social_network', True):
                current_step += 1
                self.progress_updated.emit(
                    int(current_step / total_steps * 100),
                    f"Step {current_step}/{total_steps}: Analyzing social networks..."
                )

                if self.gbi_dict is None:
                    raise ValueError("GBI data not available. Enable 'GBI Matrix Generation' first.")

                sna = SocialNetworkAnalyzer(self.config)
                node_stats, net_stats = sna.analyze_networks(
                    self.gbi_dict,
                    self.metadata_df,
                    progress_callback=lambda msg: self.progress_updated.emit(
                        int(current_step / total_steps * 100), msg
                    )
                )

                self.step_completed.emit("Social Network Analysis", "ALLTRIAL_SNA_*.csv")

            # Step 5: Edgelist Creation
            if self.enabled_steps.get('edgelist', True):
                current_step += 1
                self.progress_updated.emit(
                    int(current_step / total_steps * 100),
                    f"Step {current_step}/{total_steps}: Creating edgelist..."
                )

                if self.gbi_dict is None:
                    raise ValueError("GBI data not available. Enable 'GBI Matrix Generation' first.")

                edgelist_gen = EdgelistGenerator(self.config)
                edgelist_df = edgelist_gen.create_edgelist(
                    self.gbi_dict,
                    self.movebout_df,
                    progress_callback=lambda msg: self.progress_updated.emit(
                        int(current_step / total_steps * 100), msg
                    )
                )

                self.step_completed.emit("Edgelist Creation", "ALLTRIAL_MOVEBOUT_GBI_edgelist.csv")

            # Step 6: Displacement Detection
            if self.enabled_steps.get('displacement', True):
                current_step += 1
                self.progress_updated.emit(
                    int(current_step / total_steps * 100),
                    f"Step {current_step}/{total_steps}: Detecting displacement events..."
                )

                if self.movebout_df is None:
                    raise ValueError("Movement bout data not available. Enable 'Movement Bout Detection' first.")

                disp_detector = DisplacementDetector(self.config)
                disp_df = disp_detector.detect_displacements(
                    self.movebout_df,
                    self.metadata_df,
                    progress_callback=lambda msg: self.progress_updated.emit(
                        int(current_step / total_steps * 100), msg
                    )
                )

                self.step_completed.emit("Displacement Detection", "ALLTRIAL_MOVEBOUT_GBI_displace.csv")

            # Step 7: Hinde Index Calculation
            if self.enabled_steps.get('hinde_index', True):
                current_step += 1
                self.progress_updated.emit(
                    int(current_step / total_steps * 100),
                    f"Step {current_step}/{total_steps}: Calculating Hinde indices..."
                )

                if self.gbi_dict is None:
                    raise ValueError("GBI data not available. Enable 'GBI Matrix Generation' first.")

                hinde_calc = HindeIndexCalculator(self.config)
                broad_df, narrow_df, summary_df = hinde_calc.calculate_hinde_indices(
                    self.gbi_dict,
                    self.movebout_df,
                    self.metadata_df,
                    progress_callback=lambda msg: self.progress_updated.emit(
                        int(current_step / total_steps * 100), msg
                    )
                )

                self.step_completed.emit("Hinde Index Calculation", "ALLTRIAL_MOVEBOUT_GBI_hinde_*.csv")

            # Processing complete
            self.progress_updated.emit(100, "Processing complete!")
            self.processing_complete.emit(True, "RFID pipeline completed successfully!")

        except Exception as e:
            error_msg = f"Error during processing: {str(e)}\n\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
            self.processing_complete.emit(False, error_msg)
