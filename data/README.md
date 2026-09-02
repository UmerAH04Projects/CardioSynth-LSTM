# Data

The project uses the MIT-BIH Arrhythmia Database via WFDB/PhysioNet.

Raw database files are intentionally not committed to GitHub. Run the dataset and preprocessing cells in the notebook to download the required records.

Preprocessing includes 0.5–45 Hz bandpass filtering, Z-score normalization, annotation-based beat extraction, and fixed-length windowing.
