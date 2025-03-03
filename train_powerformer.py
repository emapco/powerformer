from typing import Literal

import hydra
import optuna
from hydra_trainer import BaseTrainer
from omegaconf import DictConfig

from powerformer.dataset import PowerformerDataset, PowerformerDatasetConfig
from powerformer.model import Powerformer, PowerformerConfig
from powerformer.util import get_device_name

# logging.getLogger("transformers").setLevel(logging.ERROR)


class PowerformerTrainer(BaseTrainer[PowerformerDataset, PowerformerDatasetConfig]):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def dataset_factory(
        self,
        dataset_cfg: PowerformerDatasetConfig,
        dataset_key: Literal["train", "eval"],
    ) -> PowerformerDataset:
        return PowerformerDataset(
            dataset_cfg,
            dataset_key,
            self.cfg.model.context_len,
            self.cfg.model.prediction_len,
        )

    def model_init_factory(self):
        def model_init(trial: optuna.Trial | None = None):
            model_cfg = self.get_trial_model_cfg(trial, self.cfg)

            cfg = PowerformerConfig(**model_cfg)
            model = (
                Powerformer.from_pretrained(self.cfg.checkpoint_path)
                if self.cfg.checkpoint_path is not None
                else Powerformer(cfg)
            )
            model.to(get_device_name())  # type: ignore
            if trial is None:
                print(model)

            print(f"Model size: {model.num_parameters()} parameters.")
            return model

        return model_init


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    predictor = PowerformerTrainer(cfg)
    predictor.train()


if __name__ == "__main__":
    main()
