# Changelog of the dataprep package

The dataprep package was introduced in S-GAIN pre-release v0.1.4-alpha. It has a separate distributions.py file
containing all distributions in J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative
Adversarial Nets", ICML, 2018. https://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf. However, the dataloader was
introduced as a utility package in S-GAIN pre-release v0.1.0-alpha and was part of the original GAIN code.
https://github.com/jsyoon0823/GAIN.

## Version 2.0

This version vastly improves on the previous dataloader.

- Now contains all missing value distributions proposed in the GAIN paper.
- Added an option to include labels in the dataset. This is for testing the viability of imputation methods as
  predictive models (classifiers/regressors).
- Import UCI datasets directly from the repository.
- Added an option for storing the downloaded datasets for offline access.
- Distributions are now independent of utils.utils.

## Version 1.1

This version was introduced in S-GAIN pre-release v0.1.3-alpha.

- Now can store prepared datasets

## Version 1.0

This version marks the original dataloader by Yoon et al.