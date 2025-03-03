# Unofficial Powerformer PyTorch Implementation

This is an unofficial PyTorch implementation of [Powerformer: A Transformer with Weighted Causal Attention for Time-Series Forecasting](https://arxiv.org/abs/2502.06151).

🚧 **Work in Progress** 🚧

## Running the Model on an Example Dataset

To train the model using the default example dataset, run:

```bash
python src/train_powerformer.py
```

## Customizing Model Parameters

Model configurations can be modified in either `trainer.yaml` or `config.yaml`. Alternatively, parameters can be overridden directly via the command line:

```bash
python train_powerformer.py model.num_layers=4
```

## Hyperparameter Tuning

Hyperparameter search spaces can be defined in `trainer.yaml` and `config.yaml`:

- `trainer.yaml` configures tuning for `transformers.TrainingArguments` parameters.
- `config.yaml` allows customization of model-specific parameters.

To enable hyperparameter optimization, run:

```bash
python train_powerformer.py do_hyperoptim=true
```

## Dataset Formatting

A Jupyter notebook (`data/data.ipynb`) is provided for dataset preparation and splitting. The formatted dataset follows a structure similar to the [Chronos dataset](https://huggingface.co/datasets/autogluon/chronos_datasets).

The initial dataset, stored as a Parquet file, should include:

- A **time** or **timestamp** column.
- Any number of numerical feature columns for forecasting.
