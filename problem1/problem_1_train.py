"""
Implement a training loop for the MLP and SIREN models.
"""

import json
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

from problem_1_gradients import gradient, laplace
from problem_1_mlp import MLP
from problem_1_siren import SIREN
from utils import ImageDataset, plot, psnr
from typing import Any, Dict, List, Optional, Tuple


def make_run_output_dir(model: str, base_dir: str = "outputs") -> str:
    """Return ``outputs/{model}_{timestamp}`` (e.g. ``outputs/siren_20250531_203045``)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(base_dir) / f"{model.lower()}_{stamp}")


def _collect_train_params(
    model: str,
    dataset: ImageDataset,
    lr: float,
    total_steps: int,
    steps_til_summary: int,
    device: torch.device,
    net: torch.nn.Module,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "model": model,
        "lr": lr,
        "total_steps": total_steps,
        "steps_til_summary": steps_til_summary,
        "device": str(device),
        "dataset_height": dataset.height,
        "in_features": 2,
        "out_features": 1,
        "loss": "mse",
        "optimizer": "adam",
        **kwargs,
    }
    if isinstance(net, MLP):
        params.update(
            {
                "hidden_features": net.hidden_features,
                "hidden_layers": net.hidden_layers,
                "bias": net.bias,
                "activation": net.activation,
            }
        )
    elif isinstance(net, SIREN):
        params.update(
            {
                "hidden_features": net.hidden_features,
                "hidden_layers": net.hidden_layers,
                "bias": net.bias,
                "last_layer_linear": net.last_layer_linear,
                "first_omega_0": net.first_omega_0,
                "hidden_omega_0": net.hidden_omega_0,
            }
        )
    return params


def train(
    model,  # "MLP" or "SIREN"
    dataset: ImageDataset,  # Dataset of coordinates and pixels for an image
    lr: float,  # Learning rate
    total_steps: int,  # Number of gradient descent step
    steps_til_summary: int,  # Number of steps between summaries (i.e. print/plot)
    device: torch.device,  # "cuda" or "cpu"
    output_dir: Optional[str] = None,  # If set, save plots and metrics here
    **kwargs: Dict[str, Any],  # Model-specific arguments
) -> Tuple[List[float], List[float]]:
    """
    Train the model on the provided dataset.

    Returns ``(loss_history, psnr_history)`` for compatibility with ``benchmark.ipynb``.
    """
    model_kwargs = dict(kwargs)
    in_features = model_kwargs.pop("in_features", 2)
    out_features = model_kwargs.pop("out_features", 1)
    if model == "MLP":
        net: torch.nn.Module = MLP(
            in_features=in_features, out_features=out_features, **model_kwargs
        )
    elif model == "SIREN":
        net = SIREN(
            in_features=in_features, out_features=out_features, **model_kwargs
        )
    else:
        raise ValueError(f"Unknown model {model!r}; expected 'MLP' or 'SIREN'.")

    train_params = _collect_train_params(
        model, dataset, lr, total_steps, steps_til_summary, device, net, kwargs
    )

    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    coords, pixels = next(iter(loader))
    coords = coords[0].to(device)
    pixels = pixels[0].to(device)

    loss_history: List[float] = []
    psnr_history: List[float] = []
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "params.json", "w") as f:
            json.dump(train_params, f, indent=2)

    def summarize(step: int) -> None:
        outputs, coords_grad = net(coords)
        loss_val = torch.nn.functional.mse_loss(outputs, pixels).item()
        psnr_val = psnr(outputs, pixels).item()
        print(f"Step {step}: MSE={loss_val:.6f}, PSNR={psnr_val:.2f} dB")

        img_grad = gradient(outputs, coords_grad)
        img_laplacian = laplace(outputs, coords_grad)
        field_path = str(out / f"field_step_{step:06d}.png") if out is not None else None
        plot(dataset, outputs, img_grad, img_laplacian, save_path=field_path)

        _, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(loss_history, label="MSE")
        axs[0].set_xlabel("Step")
        axs[0].set_ylabel("MSE")
        axs[0].set_title("Training loss")
        axs[0].legend()
        axs[1].plot(psnr_history, label="PSNR", color="C1")
        axs[1].set_xlabel("Step")
        axs[1].set_ylabel("PSNR (dB)")
        axs[1].set_title("PSNR")
        axs[1].legend()
        plt.tight_layout()
        if out is not None:
            plt.savefig(out / f"metrics_step_{step:06d}.png", dpi=150, bbox_inches="tight")
        plt.show()

    net.train()
    for step in range(total_steps):
        optimizer.zero_grad()
        outputs, _ = net(coords)
        loss = torch.nn.functional.mse_loss(outputs, pixels)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        with torch.no_grad():
            psnr_history.append(psnr(outputs, pixels).item())

        if step % steps_til_summary == 0:
            summarize(step)

    if (total_steps - 1) % steps_til_summary != 0:
        summarize(total_steps - 1)

    if out is not None:
        with open(out / "history.json", "w") as f:
            json.dump({"loss": loss_history, "psnr": psnr_history}, f, indent=2)
        torch.save(net.state_dict(), out / "model.pt")
        print(f"Saved outputs to {out.resolve()}")

    return loss_history, psnr_history
