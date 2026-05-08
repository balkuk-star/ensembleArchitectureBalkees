import json
import os
from datetime import datetime
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

from utils import compute_metrics_from_logits, ensure_binary_output

DEFAULT_OUTPUT_DIR = "/content/drive/MyDrive/ensemble_outputs"


def force_colab_inline() -> None:
    """Ensure matplotlib renders inline in notebook/Colab contexts."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("matplotlib", "inline")
    except Exception:
        # Safe no-op outside notebook environments.
        pass


def ensure_output_dir(output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_plot(fig: plt.Figure, name: str, output_dir: str = DEFAULT_OUTPUT_DIR, dpi: int = 300) -> str:
    output_dir = ensure_output_dir(output_dir)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def evaluate_model(model, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    bag = {"Dice": [], "IoU": [], "Precision": [], "Recall": [], "Accuracy": []}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            metrics = compute_metrics_from_logits(model(x), y)
            for k, v in metrics.items():
                bag[k].append(v)
    return {k: float(np.mean(v)) for k, v in bag.items()}


def plot_training_curves(history: Dict[str, list], title: str, output_dir: str = DEFAULT_OUTPUT_DIR):
    force_colab_inline()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        ax[0].plot(history["val_loss"], label="val_loss")
    ax[0].set_title(f"{title} Loss")
    ax[0].legend()

    ax[1].plot(history["val_dice"], label="val_dice")
    ax[1].set_title(f"{title} Dice")
    ax[1].legend()
    save_plot(fig, f"loss_curve_{title.lower().replace(' ', '_').replace('+', 'plus')}.png", output_dir)
    plt.show()


def _build_pred_grid(
    x: torch.Tensor,
    y: torch.Tensor,
    pred: torch.Tensor,
    model_name: str,
    num_samples: int,
):
    x_np = x.cpu().permute(0, 2, 3, 1).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x_np = np.clip(x_np * std + mean, 0, 1)
    pred_np = (pred[:, 0].numpy() > 0.5).astype(np.float32)

    rows = min(num_samples, x.shape[0])
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(rows):
        axes[i, 0].imshow(x_np[i])
        axes[i, 0].set_title("Input")
        axes[i, 1].imshow(y[i, 0].cpu().numpy(), cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pred_np[i], cmap="gray")
        axes[i, 2].set_title(f"{model_name} Prediction")
        for c in range(3):
            axes[i, c].axis("off")
    fig.tight_layout()
    return fig


def visualize_predictions(
    model: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
    loader,
    device: torch.device,
    num_samples: int = 5,
    model_name: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
):
    force_colab_inline()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    model_dict: Dict[str, torch.nn.Module]
    if isinstance(model, dict):
        model_dict = model
    else:
        label = model_name or model.__class__.__name__
        model_dict = {label: model}

    for m in model_dict.values():
        m.eval()

    with torch.no_grad():
        preds = {
            name: torch.sigmoid(ensure_binary_output(m(x), size=y.shape[2:])).cpu()
            for name, m in model_dict.items()
        }

    saved_paths = []
    for name, pred in preds.items():
        fig = _build_pred_grid(x, y, pred, name, num_samples=num_samples)
        path = save_plot(fig, f"predictions_{name.lower().replace(' ', '_').replace('+', 'plus')}.png", output_dir)
        saved_paths.append(path)
        print(f"[Saved] {path}")
        plt.show()
    return saved_paths


def _prediction_mask(pred: torch.Tensor, sample_idx: int, threshold: float = 0.5) -> np.ndarray:
    """Return one thresholded prediction mask as a NumPy array for plotting."""
    return (pred[sample_idx, 0].numpy() > threshold).astype(np.float32)


def visualize_model_comparison(
    models: Dict[str, torch.nn.Module],
    loader,
    device: torch.device,
    num_samples: int = 2,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    filename: str = "predictions_all_models.png",
):
    """
    Build, save, and display a single comparison image with columns:
    Input | Ground Truth | <model1> | <model2> | ...
    for the selected samples.
    """
    force_colab_inline()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    for model in models.values():
        model.eval()

    with torch.no_grad():
        preds = {
            name: torch.sigmoid(ensure_binary_output(model(x), size=y.shape[2:])).cpu()
            for name, model in models.items()
        }

    x_np = x.cpu().permute(0, 2, 3, 1).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x_np = np.clip(x_np * std + mean, 0, 1)

    rows = min(num_samples, x.shape[0])
    model_names = list(models.keys())
    cols = 2 + len(model_names)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.asarray(axes)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(rows):
        axes[i, 0].imshow(x_np[i])
        axes[i, 0].set_title("Input")
        axes[i, 1].imshow(y[i, 0].detach().cpu().numpy(), cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        for j, name in enumerate(model_names, start=2):
            axes[i, j].imshow(_prediction_mask(preds[name], i), cmap="gray")
            axes[i, j].set_title(name)
        for c in range(cols):
            axes[i, c].axis("off")

    fig.tight_layout()
    path = save_plot(fig, filename, output_dir=output_dir)
    print(f"[Saved] {path}")
    plt.show()
    return path


def build_comparison_table(metrics_by_model: Dict[str, Dict[str, float]], params_by_model: Dict[str, int], fps_by_model: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for name, metrics in metrics_by_model.items():
        rows.append({
            "Model": name,
            "Dice": metrics["Dice"],
            "IoU": metrics["IoU"],
            "Params": params_by_model.get(name, 0),
            "FPS": fps_by_model.get(name, 0.0),
        })
    return pd.DataFrame(rows).sort_values("Dice", ascending=False)


def save_comparison_table(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR, filename: str = "comparison_table.csv") -> str:
    output_dir = ensure_output_dir(output_dir)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    return path


def print_comparison_table(df: pd.DataFrame) -> None:
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))


def plot_metrics(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR):
    force_colab_inline()

    # Dice bar chart
    fig_dice, ax_dice = plt.subplots(figsize=(8, 4))
    ax_dice.bar(df["Model"], df["Dice"])
    ax_dice.set_ylabel("Dice")
    ax_dice.set_title("Dice Comparison")
    ax_dice.set_ylim(0, 1)
    ax_dice.tick_params(axis="x", rotation=20)
    save_plot(fig_dice, "dice_plot.png", output_dir)
    plt.show()

    # IoU bar chart
    fig_iou, ax_iou = plt.subplots(figsize=(8, 4))
    ax_iou.bar(df["Model"], df["IoU"], color="darkorange")
    ax_iou.set_ylabel("IoU")
    ax_iou.set_title("IoU Comparison")
    ax_iou.set_ylim(0, 1)
    ax_iou.tick_params(axis="x", rotation=20)
    save_plot(fig_iou, "iou_plot.png", output_dir)
    plt.show()

    # FPS vs Params (optional)
    if {"FPS", "Params"}.issubset(df.columns):
        fig_pp, ax_pp = plt.subplots(figsize=(7, 5))
        ax_pp.scatter(df["Params"], df["FPS"], s=80)
        for _, row in df.iterrows():
            ax_pp.annotate(row["Model"], (row["Params"], row["FPS"]), textcoords="offset points", xytext=(5, 5))
        ax_pp.set_xlabel("Parameters")
        ax_pp.set_ylabel("FPS")
        ax_pp.set_title("FPS vs Params")
        save_plot(fig_pp, "fps_vs_params.png", output_dir)
        plt.show()


def save_metrics_log(metrics_by_model: Dict[str, Dict[str, float]], output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    output_dir = ensure_output_dir(output_dir)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"metrics_log_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_by_model, f, indent=2)
    return path


def plot_dice_bars(metrics_by_model: Dict[str, Dict[str, float]], output_dir: str = DEFAULT_OUTPUT_DIR):
    df = build_comparison_table(metrics_by_model, {k: 0 for k in metrics_by_model}, {k: 0.0 for k in metrics_by_model})
    plot_metrics(df, output_dir=output_dir)
