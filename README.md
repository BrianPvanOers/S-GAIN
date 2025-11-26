# Codebase of S-GAIN

When using this code, please cite the following paper:

Paper: van Oers, B.P., Baysal Erez, I., van Keulen, M. (2026). Sparse GAIN: Imputation Methods to Handle Missing Values
with Sparse Initialization. In: Martínez, L., et al. Intelligent Data Engineering and Automated Learning – IDEAL 2025.
IDEAL 2025. Lecture Notes in Computer Science, vol 16238. Springer, Cham. https://doi.org/10.1007/978-3-032-10486-1_22

Authors: Brian Patrick van Oers, Işıl Baysal Erez, Maurice van Keulen

Release: This paper is associated with [pre-release v0.1.0-alpha](
https://github.com/BrianPvanOers/S-GAIN/releases/tag/v0.1.0-alpha) *

Contact: b.p.vanoers@student.utwente.nl

\* Alternatively, for easy replication, one may load the IDEAL2025 settings to run the experiments associated with this
paper, with the current tools.

---

## About the project

We adapted the original GAIN code for our work: [J. Yoon, J. Jordon and M. van der Schaar, "GAIN: Missing Data
Imputation using Generative Adversarial Nets," International Conference on Machine Learning (ICML), 2018.](
https://github.com/jsyoon0823/GAIN)

We created a framework for (automated) testing and implemented Dynamic Sparse Training strategies to improve
computational efficiency and therefore energy consumption, memory usage and imputation time, and possibly increase
performance and reduce failure rates. We plan to rebuild the model in TensorFlow 2.x using Sparse Tensors and INT8
precision, and then run it on the GPU to speed up the experiments.

We ran our experiments using python 3.11, earlier or later versions might have package conflicts.

---

## How to use s_gain.py TODO

The settings are specified in config.py. Defaults can be loaded by using ```settings load default```. The settings of
the paper are available as ```IDEAL2025```. There is also a ```showcase``` available to demonstrate the system.

```run``` will run all possible combinations of these settings.

- **input (experiments):** the folder where the experiments were saved to (optional, default: 'output')
- **output (analysis):** save the analysis to a different folder (optional, default: 'analysis')

One may use this file to run multiple experiments in sequence, automatically analyze them and if needed shutdown the
computer after wards (if no experiments will be run; auto_shutdown is ignored). The settings are given as lists.
run_experiments.py will run all possible combinations of these settings for n_runs times. Nonsense is ignored, i.e.
dense initialization with > 0% sparsity and non-dense initializations with 0% sparsity won't be run. Below you find
additional settings not already explained in prior sections:

### Example commands of subroutines TODO

#### Settings

```settings load```
```settings store```

###### Show, load, store or delete settings.

```shell
$ python s_gain.py settings store current
```

```shell
$ python s_gain.py settings load showcase  
```

```shell
$ python s_gain.py settings show
```

#### Prepare

###### Prepare datasets with missing data.

```shell
$ python s_gain.py prepare
```

#### Run

###### Run the experiments specified in config.py.

```shell
$ python s_gain.py run
```

#### Analyze

###### Analyze the completed experiments.

```shell
$ python s_gain.py analyze
```

### Explanation of settings

The first four sections are given as a list. The run subroutine will run all possible combinations of these settings.

- **Data preparation**
    - **dataset:** the dataset to use. [spam, letter, health, mnist, fashion_mnist, cifar10]
    - **miss_rate:** the probability of missing elements in the data.
    - **miss_modality:** the modality of missing data. [MCAR, MAR, MNAR, upscaler, square]
    - **seed:** the seed used to introduce missing elements in the data. (use None for a random seed)
    - **prepared_dataset_folder:** the folder to store the prepared datasets in.
    - **store_prepared_dataset:** whether to store the prepared dataset. useful for time intensive miss modalities, i.e.
      MAR, or for analysis.


- **S-GAIN**
    - **version:** the S-GAIN version. ['TFv1_FP32', 'TFv2_INT8']
    - **batch_size:** the number of samples in the mini-batch.
    - **hint_rate:** the hint probability.
    - **alpha:** the hyperparameter.
    - **iterations:** the number of training iterations.
    - **clipping:** enable clipping in D_prob.


- **Generator**
    - **generator_initialization:** the initialization strategy of the generator. [dense, normal_random (NR), magnitude,
      erdos_renyi (ER), erdos_renyi_normal_random (ERNR)*]
    - **generator_sparsity:** the probability of sparsity in the generator.
    - **generator_pruner:** the pruning strategy of the generator. [random, magnitude, None]
    - **generator_prune_rate:** the probability of pruning a non-zero weight in the generator, based on the number of
      non-zero weights at initialization.
    - **generator_prune_period:** the number of iterations before pruning the generator, after initialization or
      previous pruning.
    - **generator_regrower:** the regrowing strategy of the generator. [random, None]
    - **generator_regrow_rate:** the probability of regrowing a zero weight in the generator, based on the number of
      non-zero weights at initialization.
    - **generator_regrow_period:** the number of iterations before regrowing the generator, after initialization or
      previous regrowing.
    - **generator_strategy:** the training strategy of the generator. [None]
    - **generator_use_strategy:** use a complete training strategy for the generator, instead of separate
      initialization, pruning and regrowing strategies.


- **Discriminator**
    - **discriminator_initialization:** the initialization strategy of the discriminator. [dense, normal_random (NR),
      magnitude, erdos_renyi (ER), erdos_renyi_normal_random (ERNR)*]
    - **discriminator_sparsity:** the probability of sparsity in the discriminator.
    - **discriminator_pruner:** the pruning strategy of the discriminator. [random, magnitude, None]
    - **discriminator_prune_rate:** the probability of pruning a non-zero weight in the discriminator, based on the
      number of non-zero weights at initialization.
    - **discriminator_prune_period:** the number of iterations before pruning the discriminator, after initialization or
      previous pruning.
    - **discriminator_regrower:** the regrowing strategy of the discriminator. [random, None]
    - **discriminator_regrow_rate:** the probability of regrowing a zero weight in the discriminator, based on the
      number of non-zero weights at initialization.
    - **discriminator_regrow_period:** the number of iterations before regrowing the discriminator, after initialization
      or previous regrowing.
    - **discriminator_strategy:** the training strategy of the discriminator. [None]
    - **discriminator_use_strategy:** use a complete training strategy for the discriminator, instead of separate
      initialization, pruning and regrowing strategies.


- **Output**
    - **output_folder:** the folder to save experiments to.
    - **no_imputation:** don't save the imputed data.
    - **no_log:** turn off the logging of metrics. (disables graphs)
    - **no_graph:** don't plot the graphs after training.
    - **no_model:** don't save the trained model.


- **Monitor**
    - **enable_rmse_monitor:** enable monitoring of the RMSE.
    - **enable_imputation_time_monitor:** enable monitoring of the imputation time.
    - **enable_memory_usage_monitor:** enable monitoring of the memory usage.
    - **enable_energy_consumption_monitor:** enable monitoring of the energy consumption.
    - **enable_sparsity_monitor:** enable monitoring of the sparsity of both models.
    - **enable_FLOPs_monitor:** enable monitoring of the FLOPs of both models. (takes significantly more time)
    - **enable_loss_monitor:** enable monitoring of the losses (cross entropy and MSE).


- **Run**
    - **n_runs:** the number of times each experiment should be performed.
    - **retry_failed_experiment:** retry a failed experiment until it successfully completes n_runs times or reaches
      max_failed_experiments.
    - **max_failed_experiments:** the maximum number of times the experiment can fail. used to prevent infinite loops.
    - **ignore_existing_files:** ignore the existing files in the output folder. (disables retry_failed_experiments, a
      random seed will always ignore existing files)


- **Analysis**
    - **analysis_folder:** the folder to save the analysis to.
    - **perform_analysis:** automatically analyze the experiments after completion.
    - **compile_metrics:** compile the metrics.
    - **plot_rmse:** plot the RMSE graphs.
    - **plot_success_rate:** plot the success rate graphs.
    - **plot_imputation_time:** plot the imputation time graphs.
    - **plot_memory_usage:** plot the memory usage graphs.
    - **plot_energy_consumption:** plot the energy consumption graphs.


- **Inclusions**
    - **inclusions:** an inclusion is a dictionary of settings. it overwrites the base config and adds the newly
      specified experiments. the config reloads before each inclusion and previously made changes don't carry over. this
      ensures each inclusion is independent of any previous inclusion.


- **Exclusions**
    - **exclusions:** an exclusion is a dictionary of settings. it removes experiments with this combination of
      settings. it overwrites the inclusions. each exclusion is independent of any previous exclusion.


- **Options**
    - **verbose:** enable verbose output to the console.
    - **no_system_information:** don't log system information.
    - **auto_shutdown:** automatically shutdown the computer after running the experiments and performing the analyses.

\* previously referred to as erdos-renyi random weight (ERRW), this was changed due to the fact that the random weights
can be picked from either a normal or a uniform distribution.

---

## Folders and files

- **datasets:** contains (some of) the datasets to run the S-GAIN imputer on. a dataset must be complete, have a header
  and the labels and index must be removed. these datasets serve as x_train. (todo: test with labels to test its
  classifier performance)
    - **feature_added:** contains datasets which have (statistical) features added to aid in assessing their importance.
    - **prepared:** contains the prepared datasets.
    - **private:** contains datasets with sensitive information, that should not be pushed to GitHub.
    - **health.csv:** Ahmed, M. (2020). Maternal Health Risk [Dataset]. UCI Machine Learning Repository.
      https://doi.org/10.24432/C5DP5D.
    - **letter.csv:** Slate, D. (1991). Letter Recognition [Dataset]. UCI Machine Learning Repository.
      https://doi.org/10.24432/C5ZP40.
    - **spam.csv:** Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). Spambase [Dataset]. UCI Machine
      Learning Repository. https://doi.org/10.24432/C53G6X.
- **models:** contains the different models.
    - **s_gain_TFv2_INT8.py:** the S-GAIN imputer. (this version uses TensorFlow 2.x and INT8 precision)
    - **s_gain_TFv1_FP32.py:** the S-GAIN imputer. (this version uses TensorFlow 1.x and FP32 precision)
- **monitors:** contains the monitor.
    - **monitor.py:** used for measuring things.
- **output:** the output folder for the experiments.
    - **[experiment].csv:** the imputed data for the experiment.
    - **[experiment]_graphs.png:** a single png file containing the graphs of: RMSE, imputation time, memory usage,
      energy consumption, sparsity, FLOPs and loss (cross entropy and MSE).
    - **[experiment]_log.json:** a log file of measurements taken throughout the experiment.
    - **[experiment]_model.json:** the trained model.
- **temp:** contains temporary files.
    - **exp_bins:** contains binary files. (used for logging measurements throughout the experiment)
    - **run_config:** the config file contains all settings used in the S-GAIN testing framework. more detail is
      provided in the explanation of settings section.
    - **sys_info.json:** caches the system information.
- **utils:** contains different utility files.
    - **flops:** contains code to calculate FLOPs. (copied from Google Research)
    - **inits:** contains files for initialization strategies.
        - **s_gain_TFv2_INT8.py:** contains the different initialization strategies for the s_gain_TFv2_INT8 version.
        - **s_gain_TFv1_FP32.py:** contains the different initialization strategies for the s_gain_TFv1_FP32 version.
    - **pruners:** contains files for pruning strategies.
        - **s_gain_TFv2_INT8.py:** contains the different pruning strategies for the s_gain_TFv2_INT8 version.
        - **s_gain_TFv1_FP32.py:** contains the different pruning strategies for the s_gain_TFv1_FP32 version.
    - **regrowers:** contains files for regrowing strategies.
        - **s_gain_TFv2_INT8.py:** contains the different regrowing strategies for the s_gain_TFv2_INT8 version.
        - **s_gain_TFv1_FP32.py:** contains the different regrowing strategies for the s_gain_TFv1_FP32 version.
    - **strategies:** contains files for advanced training strategies.
        - **s_gain_TFv2_INT8.py:** contains the different advanced training strategies for the s_gain_TFv2_INT8 version.
        - **s_gain_TFv1_FP32.py:** contains the different advanced training strategies for the s_gain_TFv1_FP32 version.
    - **analysis.py:** contains functions to analyze the experiments.
    - **data_loader.py:** loads the datasets and introduces missingness in the data.
    - **graphs2.py:** an updated version of graphs.py: plot the relevant graphs to the same file.
    - **load_store.py:** loads and stores files.
    - **metrics.py:** calculates the relevant metrics.
    - **miss_modalities:** contains the different miss modalities for the data loader.
    - **standardizers.py:** standardizes the different settings.
    - **subroutines.py:** contains the different subroutines for s_gain.py.
    - **utils.py:** contains other utilities.
- **config.py:** this file contains the settings for S-GAIN and the testing framework.
- **main.py:** the main file from which an experiment is run.
- **s_gain.py:** the main interface to interact with S-GAIN.

---

## Help support my research

If you like this work, and you would like to support its development, please consider donating:

**paypal.me/BrianPvanOers**
