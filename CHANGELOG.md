# Changelog

This file marks the changes since the submission of the manuscript to the IDEAL 2025 conference.

## Pre-release v0.2.0-beta (future)

This version implements S-GAIN in TensorFlow v2, uses INT8 precision for quantization, and uses sparse tensors (COO
matrices).

## Pre-release v0.1.5-beta (TBD)

This version is currently being developed and seeks to implement full monitoring of system resources.

- Memory usage.
- Energy consumption (possible integration with HWMonitor on Windows to detect power draw).
- CPU utilization.
- CPU temperatures (to check for throttling).
- GPU utilization.
- GPU temperatures (to check for throttling).
- GPU VRAM usage.

## Pre-release v0.1.4-beta (in development)

This version implements Statistical Feature Addition (SFA).


## Pre-release v0.1.3-alpha (in development)

This version saw a major overhaul of the testing framework and major improvements to the analysis. This version marks
the end of my role as project supervisor for Missing Data and Dynamic Sparse Training in the Data Science and Artificial
Intelligence elective at the University of Twente. It unifies the work of my students with my own work, and includes the
work of my own Research Project.

- Switched to a config based project structure.
- New dataprep package with upgraded dataloader (more details are provided in the package changelog):
  - Implements MAR and MNAR distributions.
  - Directly imports datasets from the UCI repository.
  - Returns the header and optionally the labels.
  - Can download the datasets for offline access.
  - Distributions are now independent of utils.utils.
  - Includes readme and changelog.
- Implemented pruners, regrowers, and complete training strategies.
- Implemented imputation time analysis.
- Immediately terminate failed experiments to speed up testing.
- Implemented auto shutdown for s_gain.py run.
- Improvements to the analysis. TODO more specificity.
- retry_failed_experiments now reruns the individual experiment until completion or reaching the failure instead of
  running other experiments first.
- Can now set max_failed_experiments to prevent infinite loops.
- Other improvements to the project structure.

## Pre-release v0.1.2-alpha (11-09-2025)

This version saw major improvements to the analysis. This version marks the start of my role as project supervisor for
the Data Science and Artificial Intelligence elective at the University of Twente.

- Implemented loss monitoring (cross entropy and MSE).
- Now plots all graphs to a single file along with experiment information and system details.
- Plot sizing is now consistent for all number of subplots.
- Added verbosity to analyze.py and log_and_graphs.py.

## Pre-release v0.1.1-alpha (28-08-2025)

This version saw a major overhaul of the testing framework.

- Now specifies all experiment settings for the output files for easy manipulation of data and replication of results.
- Implemented imputation time monitoring.
- run_experiments.py replaces loop_main.py and solves the issue of TensorFlow not restarting.
- Converted the jupyter notebook to pure python and allow for automatic analysis.
- Restructured the project to improve comprehension.
- Significantly improved README.

## Pre-release v0.1.0-alpha (25-08-2025)

This version is associated with: B.P. van Oers, I. Baysal Erez, M. van Keulen, "Sparse GAIN: Imputation Methods to
Handle Missing Values with Sparse Initialization", IDEAL conference, 2025. And marks the beginning of the changelog.