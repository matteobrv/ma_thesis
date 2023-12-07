import pandas as pd

from evaluator import CrossEvaluator
from pathlib import Path
from data import CustomDataset
from sklearn.model_selection import train_test_split
from typing import Any, Dict
from tune_parameters import tune_parameters
from config_parameters import double_linear_probe, single_linear_probe

PROBES_CONFIGS = [single_linear_probe, double_linear_probe]


class Experiment:
    def __init__(self, exp_name: str, exp_config: Dict[str, Any]) -> None:
        self._exp_name = exp_name
        self._lm_type = exp_config["model_type"]
        self._lm = exp_config["model"]
        # e.g. if exp_config["last_state"] = -1 -> range(-1, -2, -1) -> -1
        # if exp_config["last_state"] = -12 -> all layers are considered
        self._hidden_states = [s for s in range(-1, exp_config["last_state"]-1, -1)]
        self._probe_dim_input = exp_config["dim_input"]
        self._probe_dim_output = exp_config["dim_output"]
        self._data_sources = exp_config["data_sources"]
        self._tune_epochs = exp_config["tune_epochs"]
        self._train_epochs = exp_config["train_epochs"]
        self._tune_samples = exp_config["samples"]
        self._num_cv_k = exp_config["cv_k"]

    def run(self) -> None:
        for data in self._data_sources:
            data_name = data.split("/")[0]
            df = pd.read_csv(Path(__file__).parent / "datasets" / data)

            for state in self._hidden_states:
                # keep the same train and validation splits for each tuning experiment
                train_data, val_data = train_test_split(
                    df, train_size=0.8, shuffle=True, random_state=42
                )

                # build datasets on embedding representations from current hidden state
                dataset = CustomDataset(df, self._lm, self._lm_type, state)
                train_set = CustomDataset(train_data, self._lm, self._lm_type, state)
                val_set = CustomDataset(val_data, self._lm, self._lm_type, state)

                # tune, train, and test each probe
                for config in PROBES_CONFIGS:
                    config["dim_input"] = self._probe_dim_input
                    config["dim_output"] = self._probe_dim_output
                    config["tune_epochs"] = self._tune_epochs
                    config["train_epochs"] = self._train_epochs
                    config["samples"] = self._tune_samples
                    config["cv_k"] = self._num_cv_k

                    tuned = tune_parameters(
                        self._exp_name, data_name, config, train_set, val_set, state
                    )

                    CrossEvaluator(
                        tuned, config, dataset, self._exp_name, state, data_name
                    ).cross_evaluate()
