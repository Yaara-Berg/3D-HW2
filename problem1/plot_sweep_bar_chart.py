"""Regenerate outputs/sweeps_bar_chart.png with short labels under each run id."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUTS = Path(__file__).resolve().parent / "outputs"
OUT_PATH = OUTPUTS / "sweeps_bar_chart.png"

# Short labels (1–2 words) from problem_1_short_answers.md sweep tables
MLP_RUNS = [
    ("M1", "Baseline"),
    ("M2", "Wider"),
    ("M3", "Deeper"),
    ("M4", "Tanh"),
    ("M5", "LR high"),
    ("M6", "LR low"),
]

SIREN_RUNS = [
    ("S1", "Baseline"),
    ("S2", "Sine last"),
    ("S3", "Low ω"),
    ("S4", "High ω"),
    ("S5", "ω+sine"),
    ("S6", "Split ω"),
    ("S7", "Wider"),
    ("S8", "LR low"),
    ("S9", "LR high"),
    ("S10", "Mid ω"),
]


def _load_psnr(run_id: str, prefix: str) -> float | None:
    pattern = f"{prefix}_{run_id.lower()}_*"
    dirs = sorted(OUTPUTS.glob(pattern))
    if not dirs:
        return None
    history = json.loads((dirs[-1] / "history.json").read_text())
    return float(history["psnr"][-1])


def _x_labels(run_specs: list[tuple[str, str]]) -> list[str]:
    return [f"{run_id}\n{label}" for run_id, label in run_specs]


def _plot_panel(ax, run_specs: list[tuple[str, str]], prefix: str, title: str, color: str) -> None:
    ids = [r[0] for r in run_specs]
    psnrs = []
    for run_id, _ in run_specs:
        psnr = _load_psnr(run_id, prefix)
        if psnr is None:
            raise FileNotFoundError(f"No outputs for {prefix}_{run_id.lower()}_* under {OUTPUTS}")
        psnrs.append(psnr)

    bars = ax.bar(_x_labels(run_specs), psnrs, color=color, edgecolor="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("Final PSNR (dB)")
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.tick_params(axis="x", labelsize=8)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    _plot_panel(axes[0], MLP_RUNS, "mlp", "MLP sweep", "C0")
    _plot_panel(axes[1], SIREN_RUNS, "siren", "SIREN sweep", "C1")
    plt.tight_layout()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
