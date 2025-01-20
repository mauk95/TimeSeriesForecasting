# time_series_inergy

## Installation

Please refer to `./install.sh` for installation commands, or

Create the conda environment using the `./environment.yml` file, by running the following command: `conda env create -f environment.yml`

## Notebooks/Scripts
```
└───TimesNet _single_sample.ipynb: analyzes the ./data/Single_sample.csv data and performs modelling using N-HITS, NBEATS and TimesNet models.
└───TimesNet _multiple_sample.ipynb: analyzes the ./data/Multiple_sample_series.csv data and performs modelling using N-HITS, NBEATS and TimesNet models.
└───TimesNet _single_sample_hyperparam_tuning.py: Performs hyperparameter tuning on the N-HITS, NBEATS and TimesNet models for the single series dataset.
└───TimesNet _multiple_sample_hyperparam_tuning.py: Performs hyperparameter tuning on the N-HITS, NBEATS and TimesNet models for the multiple series dataset.
└───neural_forecast_many_models_multi_sample.py: analyzes the ./data/Multiple_sample.csv data and performs modelling using multiple models (except RNN-based, work for only step_size=1) from the NeuralForecast library.
```