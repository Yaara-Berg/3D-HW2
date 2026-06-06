"""Run MLP and SIREN training on CPU and save artifacts under problem1/outputs/."""

import torch
from utils import ImageDataset, set_seed
from problem_1_train import make_run_output_dir, train

DATASET_HEIGHT = 64
TOTAL_STEPS = 2000
STEPS_TIL_SUMMARY = 500


def main() -> None:
    set_seed(0)
    device = torch.device("cpu")
    dataset = ImageDataset(height=DATASET_HEIGHT)

    train(
        "MLP",
        dataset,
        lr=1e-3,
        total_steps=TOTAL_STEPS,
        steps_til_summary=STEPS_TIL_SUMMARY,
        device=device,
        output_dir=make_run_output_dir("MLP"),
        hidden_features=256,
        hidden_layers=4,
        activation="ReLU",
    )

    train(
        "SIREN",
        dataset,
        lr=1e-4,
        total_steps=TOTAL_STEPS,
        steps_til_summary=STEPS_TIL_SUMMARY,
        device=device,
        output_dir=make_run_output_dir("SIREN"),
        hidden_features=256,
        hidden_layers=3,
        last_layer_linear=True,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
    )


if __name__ == "__main__":
    main()
