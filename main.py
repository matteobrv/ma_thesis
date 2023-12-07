import json
import ray
import typer

from experiment import Experiment

# set num_gpus according to your hardware setup
ray.init(num_gpus=2, include_dashboard=False)


def main(experiments_json: str) -> None:
    with open(experiments_json, "r") as f:
        experiments_configs = json.load(f)

    for exp_name, exp_config in experiments_configs.items():
        Experiment(exp_name, exp_config).run()


if __name__ == "__main__":
    typer.run(main)
