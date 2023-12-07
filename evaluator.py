from sklearn.decomposition import PCA
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ray.air.result import Result
from sklearn.model_selection import KFold
from torch.nn import CosineSimilarity, MSELoss, PairwiseDistance
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset

from data import CustomDataset
from models.probes import DoubleLinearProbe, SingleLinearProbe
from tqdm import tqdm


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CrossEvaluator:
    def __init__(
        self,
        tuned_params: Result,
        config: Dict[str, Any],
        dataset: CustomDataset,
        exp_name: str,
        state: int,
        data_name: str,
    ) -> None:
        self._config = config
        self._dataset = dataset
        self._data_name = data_name
        self._splits = KFold(
            n_splits=self._config["cv_k"], shuffle=True, random_state=42
        )
        self._tuned_params = self._get_config(tuned_params)
        self._cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
        self._euclid_dis = PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
        self._state = state
        self._exp_name = exp_name

    def _get_config(self, tuned_results: Result) -> Dict[str, Any]:
        if tuned_results.config:
            return tuned_results.config
        else:
            raise ValueError("No tuned hyper-parameters available.")

    def _get_model(self, config: Dict[str, Any]) -> nn.Module:
        if config["id"] == "single_linear_probe":
            return SingleLinearProbe(
                config["dim_input"]
            )
        elif config["id"] == "double_linear_probe":
            return DoubleLinearProbe(
                config["dim_input"],
                config["dim_output"],
            )
        else:
            raise ValueError(f"{self._config['model']} is not supported")

    def _get_optimizer(self, model: nn.Module) -> Optimizer:
        return AdamW(
            params=model.parameters(),
            lr=self._tuned_params["optimizer"]["AdamW"]["lr"],
            betas=(
                self._tuned_params["optimizer"]["AdamW"]["beta1"],
                self._tuned_params["optimizer"]["AdamW"]["beta2"],
            ),
            weight_decay=self._tuned_params["optimizer"]["AdamW"]["weight_decay"],
        )

    @torch.no_grad()
    def _compute_similarity(
        self, batch_embeddings: List[torch.Tensor], measure: str, model: nn.Module
    ) -> torch.Tensor:
        model.eval()

        # retrieve X embeddings and run them through the model
        x = batch_embeddings[0].to(DEVICE)
        pred = model(x)

        if measure == "cosine-similarity":
            similarity = self._cosine_sim
        elif measure == "euclidean-distance":
            similarity = self._euclid_dis
        else:
            raise ValueError(f"{measure} is not supported")

        # each row stores the similarities for the sentences (a, b, c, d) in a
        # given sample: combinations([a, b, c, d], 2) -> ab, ac, ad, bc, bd, cd
        similarities = torch.stack(
            [
                similarity(e1, e2)
                # *batch_embeddings[1:]: all tensors except the first one i.e.
                # the original x tensor embedding from which we got pred
                for e1, e2 in combinations([pred, *batch_embeddings[1:]], 2)
            ]
        ).T

        return similarities

    def _store_as_dataframe(
        self, sents: List[Tuple], sims: Union[torch.Tensor, None]
    ) -> pd.DataFrame:
        df_sent = pd.DataFrame(sents)
        if sims is None:
            raise ValueError("No similaritis have been stored.")
        else:
            df_sims = pd.DataFrame(sims.cpu().numpy())

        df = pd.concat([df_sent, df_sims], axis=1)
        headers_sent = [f"sent_{i}" for i in range(len(sents[0]))]

        # assumption: there are 6 similarity measures in the resulting dataframe
        headers_sims = [f"pred-Y", "pred-d1", "pred-d2", "Y-d1", "Y-d2", "d1-d2"]
        df.columns = [*headers_sent, *headers_sims]

        return df

    def _train(
        self,
        train_loader: DataLoader,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: MSELoss
    ) -> None:
        model.train()

        for batch in train_loader:
            input_embedding = batch.get("x").to(DEVICE)
            target_embedding = batch.get("y").to(DEVICE)

            out = model(input_embedding)
            batch_loss = criterion(out, target_embedding)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

    def _test(self, test_loader: DataLoader, fold: int, model: nn.Module) -> None:
        for measure in ["cosine-similarity", "euclidean-distance"]:
            all_sents = []
            all_sims = None

            for batch in test_loader:
                # retrieve and store sentences pertaining to a single sample
                all_sents.extend([sample for sample in zip(*batch["sample"])])
                sims = self._compute_similarity(
                    [batch["x"], batch["y"], batch["d1"], batch["d2"]], measure, model
                )
                all_sims = sims if all_sims is None else torch.cat([all_sims, sims])

            res_dir = (
                Path(__file__).parent
                / f"results_{self._exp_name}_{self._data_name}"
                / f"{model}_{self._state}_{measure}"
            )
            res_dir.mkdir(parents=True, exist_ok=True)

            df = self._store_as_dataframe(all_sents, all_sims)
            csv_name = (
                f"{self._exp_name}_{model}_{measure}_state[{self._state}]_fold[{fold}]"
            )
            df.to_csv(res_dir / f"{csv_name}.csv", index=False)

    @torch.no_grad()
    def _get_low_dim_embeddings(
        self, test_loader: DataLoader, model: nn.Module
    ) -> pd.DataFrame:
        model.eval()
        fold_embeddings = np.array([]).reshape(0, self._config["dim_output"])
        labels = []

        for batch in test_loader:
            pred_x = model(batch["x"]).detach()
            pred_x = pred_x.view(-1, self._config["dim_output"]).cpu().numpy()
            fold_embeddings = np.vstack([fold_embeddings, pred_x])
            labels.extend([0 for _ in range(len(batch["x"]))])

            batch_y = batch["y"].view(-1, self._config["dim_output"]).cpu().numpy()
            fold_embeddings = np.vstack([fold_embeddings, batch_y])
            labels.extend([1 for _ in range(len(batch["y"]))])

            batch_d1 = batch["d1"].view(-1, self._config["dim_output"]).cpu().numpy()
            fold_embeddings = np.vstack([fold_embeddings, batch_d1])
            labels.extend([2 for _ in range(len(batch["d1"]))])

            batch_d2 = batch["d2"].view(-1, self._config["dim_output"]).cpu().numpy()
            fold_embeddings = np.vstack([fold_embeddings, batch_d2])
            labels.extend([3 for _ in range(len(batch["d2"]))])

        fold_embeddings = PCA(n_components=2).fit_transform(fold_embeddings)
        labels = np.asarray(labels).reshape(-1, 1)

        df = pd.DataFrame(
            np.hstack([fold_embeddings, labels]), columns=["x", "y", "labels"]
        )

        return df

    def cross_evaluate(self) -> None:
        dataset_idx = np.arange(len(self._dataset))
        splits_indices = list(self._splits.split(dataset_idx))

        for fold, (train_idx, test_idx) in enumerate((pbar := (tqdm(splits_indices)))):
            # get a new model, loss and optimizer for each fold
            model = self._get_model(self._config).to(DEVICE)
            criterion = MSELoss()
            optimizer = self._get_optimizer(model)

            pbar.set_description(
                (
                    f"\n{self._exp_name} -- evaluate {model} on {self._data_name}"
                    f" | state: {self._state} | fold: {fold}"
                )
            )
            train_loader = DataLoader(
                dataset=Subset(indices=train_idx.tolist(), dataset=self._dataset),
                batch_size=self._tuned_params["batch_size"],
            )
            test_loader = DataLoader(
                dataset=Subset(indices=test_idx.tolist(), dataset=self._dataset),
                batch_size=self._tuned_params["batch_size"],
            )

            for _ in range(self._config["train_epochs"]):
                self._train(train_loader, model, optimizer, criterion)

            # evaluate on current test fold
            self._test(test_loader, fold, model)

            # store fold embeddings
            fold_embeddings = self._get_low_dim_embeddings(test_loader, model)
            (Path(__file__).parent / f"embeddings_{self._exp_name}").mkdir(exist_ok=True)
            fold_embeddings.to_csv(Path(__file__).parent / 
                f"embeddings_{self._exp_name}" / 
                f"{self._exp_name}_{self._config['id']}_state[{self._state}]_fold[{fold}].csv"
            )
