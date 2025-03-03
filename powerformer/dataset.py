import contextlib
from typing import Literal

import datasets
import numpy as np
import torch
import torch.nn.functional as F
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

        if cfg.date_col is not None:
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

    def _process_batches(self, feature) -> tuple[torch.Tensor, torch.Tensor]:
        padded_vals, masks = [], []
        for batch_item in feature:
            padded, mask = self.pad_and_mask(
                batch_item, self.context_len + self.prediction_len
            )
            padded_vals.append(padded)
            masks.append(mask)
        return torch.stack(padded_vals), torch.stack(masks)

    def _process_features(self, row):
        feat_list, mask_list = [], []
        for _, features in row.items():
            padded_vals, masks = self._process_batches(
                features
            )  # shape: (batch, context_len+prediction_len)
            # Stack padded tensors for each feature dimension
            feat_list.append(padded_vals)
            mask_list.append(masks)

        return (
            torch.stack(feat_list),
            torch.stack(mask_list),
        )

    def __getitem__(self, idx):
        row = self._dataset[idx]
        feats, feat_mask = self._process_features(row)  # F B S

        rearrange_dim = "f b s -> b f s"  # powerformer expected data arrangement
        feats = rearrange(feats, rearrange_dim)
        feat_mask = rearrange(feat_mask, rearrange_dim)
        return {
            "context": feats[:, :, : -self.prediction_len],
            "mask": feat_mask[:, :, : -self.prediction_len],
            "labels": feats[:, :, -self.prediction_len :],
        }

    @staticmethod
    def pad_and_mask(tensor: torch.Tensor, target_length: int, pad_value=0):
        # Pads tensor to target_length and creates a boolean mask (1: data, 0: pad)
        curr_length = tensor.size(-1)
        if curr_length < target_length:
            pad_size = target_length - curr_length
            padded = F.pad(tensor, (0, pad_size), value=pad_value)
            mask = torch.cat(
                [
                    torch.ones(curr_length, dtype=torch.bool),
                    torch.zeros(pad_size, dtype=torch.bool),
                ]
            )
        else:
            padded = tensor[..., :target_length]
            mask = torch.ones(target_length, dtype=torch.bool)
        return padded, mask
