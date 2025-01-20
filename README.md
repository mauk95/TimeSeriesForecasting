# TimeSeriesForecasting

This repository uses Nixtla's NeuralForecast library for Single and Multi-timseries forecasting.

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

## Citation
```
@misc{olivares2022library_neuralforecast,
    author={Kin G. Olivares and
            Cristian Challú and
            Azul Garza and
            Max Mergenthaler Canseco and
            Artur Dubrawski},
    title = {{NeuralForecast}: User friendly state-of-the-art neural forecasting models.},
    year={2022},
    howpublished={{PyCon} Salt Lake City, Utah, US 2022},
    url={https://github.com/Nixtla/neuralforecast}
}
```