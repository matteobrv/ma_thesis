from ray import tune


single_linear_probe = {
    "id": "single_linear_probe",
    "optimizer": {
        "AdamW": {
            "lr": tune.loguniform(1e-7, 1e-4),
            "weight_decay":tune.loguniform(1e-3, 1e-2),
            "beta1": tune.loguniform(5e-1, 9e-1),
            "beta2": tune.loguniform(5e-1, 9e-1),
        }
    },
    "batch_size": tune.choice([8, 16, 32])
}

double_linear_probe = {
    "id": "double_linear_probe",
    "optimizer": {
        "AdamW": {
            "lr": tune.loguniform(1e-7, 1e-4),
            "weight_decay":tune.loguniform(1e-3, 1e-2),
            "beta1": tune.loguniform(5e-1, 9e-1),
            "beta2": tune.loguniform(5e-1, 9e-1),
        }
    },
    "batch_size": tune.choice([8, 16, 32])
}
