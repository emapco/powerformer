import contextlib
from typing import Literal

import datasets
import numpy as np
import torch
from einops import rearrange
from hydra_trainer import BaseDataset

from .types import PowerformerDatasetConfig


class PowerformerDataset(BaseDataset):
    def __init__(
        self,
        cfg: PowerformerDatasetConfig,
        dataset_key: Literal["train", "eval"],
        context_length: int,
        prediction_length: int,
    ):
        self.cfg = cfg

        dataset = datasets.load_dataset(
            "parquet",
            data_files=[cfg.train_path if dataset_key == "train" else cfg.eval_path],
            split="train",
        )
        assert isinstance(dataset, datasets.arrow_dataset.Dataset)

        if cfg.sample_size:
            dataset = dataset.select(range(min(cfg.sample_size, len(dataset))))

        if hasattr(cfg, "feature_cols") and cfg.feature_cols is not None:
            feature_columns = list(cfg.feature_cols)
        else:
            feature_columns = list(dataset.column_names)

        if hasattr(cfg, "date_col") and cfg.date_col is not None:
            with contextlib.suppress(ValueError):
                feature_columns.remove(cfg.date_col)

        # Sort columns for consistent feature positioning and set date_col as the last feature for later processing
        sorted_feature_columns = sorted(feature_columns)
        self._dataset = dataset.select_columns(sorted_feature_columns)
        self._dataset.set_format(type="torch", columns=sorted_feature_columns)

        self.feature_len = len(sorted_feature_columns)
        self.context_len = context_length
        self.prediction_len = prediction_length

    def convert_timestamp_to_iso(self, batch):
        batch[self.cfg.date_col] = [
            [np.datetime64(dt, "s").astype(np.int64) for dt in sublist]
            for sublist in batch[self.cfg.date_col]
        ]
        return batch

    def __len__(self) -> int:
        return len(self._dataset)

    def _process_features(self, row: dict):
        feat_list = [features for features in row.values()]
        return torch.stack(feat_list)

    def __getitem__(self, idx):
        row = self._dataset[idx]
        feats = rearrange(
            self._process_features(row), "f b s -> b f s"
        )  # powerformer expected data arrangement
        return {
            "context": feats[:, :, : -self.prediction_len],
            "labels": feats[:, :, -self.prediction_len :],
        }
