import torch
import torch.nn as nn

from models.probes import SingleLinearProbe, DoubleLinearProbe
from pathlib import Path
from ray.tune import Trainable
from typing import Any, Dict, Optional, Union
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from data import CustomDataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer(Trainable):
    """https://docs.ray.io/en/latest/tune/api/trainable.html#class-api-checkpointing"""
    def setup(self, config: Dict[str, Any], train_data: CustomDataset, val_data: CustomDataset):
        self._train_data = train_data
        self._config = config
        self._val_data = val_data
        self._model = self._get_model(self._config).to(DEVICE)
        self._optimizer = self._get_optimizer(self._config)
        self._train_loader = DataLoader(
            self._train_data, self._config["batch_size"], shuffle=True
        )
        self._val_loader = DataLoader(
            self._val_data, self._config["batch_size"], shuffle=True
        )

    def _get_model(self, config: Dict[str, Any]) -> nn.Module:
        if config["id"] == "single_linear_probe":
            return SingleLinearProbe(
                config["dim_input"]
            )
        elif config["id"] == "double_linear_probe":
            return DoubleLinearProbe(
                config["dim_input"],
                config["dim_output"]
            )
        else:
            raise ValueError(f"{config['model']} is not supported")

    def _get_optimizer(self, config) -> Optimizer:
        return AdamW(
            params=self._model.parameters(),
            lr=config["optimizer"]["AdamW"]["lr"],
            betas=(
                config["optimizer"]["AdamW"]["beta1"],
                config["optimizer"]["AdamW"]["beta2"],
            ),
            weight_decay=config["optimizer"]["AdamW"]["weight_decay"],
        )

    def step(self):
        avg_val_loss = self._validate()
        avg_train_loss = self._train()

        return {"val_loss": avg_val_loss, "train_loss": avg_train_loss}

    def _train(self):
        self._model.train()
        criterion = MSELoss()

        # sum of all batch losses for a given epoch
        train_loss = 0

        for batch in self._train_loader:
            input_embedding = batch.get("x").to(DEVICE)
            target_embedding = batch.get("y").to(DEVICE)
            
            out = self._model(input_embedding)
            batch_loss = criterion(out, target_embedding)
            train_loss += batch_loss.item() * input_embedding.size(0)

            self._optimizer.zero_grad()
            batch_loss.backward()

            self._optimizer.step()
        
        # compute average sample loss for a given epoch
        # https://discuss.pytorch.org/t/on-running-loss-and-average-loss/107890
        # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#training-the-model
        avg_train_loss = train_loss / len(list(self._train_loader.sampler))

        return avg_train_loss

    @torch.no_grad()
    def _validate(self):
        self._model.eval()
        criterion = MSELoss()

        # sum of all batch losses for a given epoch
        val_loss = 0

        for batch in self._val_loader:
            input_embedding = batch.get("x").to(DEVICE)
            target_embedding = batch.get("y").to(DEVICE)

            out = self._model(input_embedding)
            batch_loss = criterion(out, target_embedding)
            val_loss += batch_loss.item() * input_embedding.size(0)

        # compute average sample loss for a given epoch
        avg_val_loss = val_loss / len(list(self._val_loader.sampler))

        return avg_val_loss

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        checkpoint_path = str(Path(checkpoint_dir) / f"{self._model}.pt")
        torch.save(self._model.state_dict(), checkpoint_path)

        return checkpoint_dir
