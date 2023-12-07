from pathlib import Path
from typing import Any, Dict

import json
import typer
from ray.air import CheckpointConfig, RunConfig
from ray.air.result import Result
from ray.tune import TuneConfig, Tuner, with_parameters, with_resources
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import Dataset

from trainer import Trainer


def tune_parameters(
    exp_name: str,
    data_name: str,
    config: Dict[str, Any],
    train_data: Dataset,
    val_data: Dataset,
    state: int
) -> Result:

    # name for the current set of trials
    experiment_name = f"{exp_name}_{config['id']}_{state}_{data_name}"

    # set up and run tuning for current model and config
    my_checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    )

    my_run_config = RunConfig(
        stop={"training_iteration": config["tune_epochs"]},
        checkpoint_config=my_checkpoint_config,
        name=experiment_name,
        local_dir=str(Path(__file__).parent / f"checkpoints_{exp_name}_{data_name}"),
        verbose=0,
    )

    asha_scheduler = ASHAScheduler(
        time_attr="training_iteration",
        grace_period=3,
        reduction_factor=3,
        brackets=1,
    )

    my_tune_config = TuneConfig(
        mode="min",
        metric="val_loss",
        scheduler=asha_scheduler,
        num_samples=config["samples"],
        max_concurrent_trials=None
    )

    tuner = Tuner(
        trainable=with_resources(
            with_parameters(Trainer, train_data=train_data, val_data=val_data),
            # see: https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html
            {"cpu": 1, "gpu": 0.2}
        ),
        run_config=my_run_config,
        tune_config=my_tune_config,
        param_space=config,
    )

    # retrieve best experiment results
    results = tuner.fit().get_best_result()
    
    # store best parameters and path to model checkpoint
    with open(Path(__file__).parent / f"tuning_results_{experiment_name}.jsonl", "a") as f:
        best_params = results.config
        json.dump(best_params, f)
        f.write("\n")

    return results


if __name__ == "__main__":
    typer.run(tune_parameters)
