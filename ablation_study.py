"""Leave-one-model-out ablation study for the segmentation ensemble.

This script is intentionally standalone: it reuses the existing dataset, model,
evaluation, and utility modules without modifying the current training pipeline.
It trains one lightweight weighted-fusion head for the full ensemble and one head
for each ensemble variant with one base model removed, then exports metrics,
parameter counts, FPS, experiment parameters, and side-by-side prediction images.

Example:
    python ablation_study.py \
        --data_dir /content/Kvasir-SEG \
        --resunet_source resunet++_kvasir.py \
        --resunet_ckpt best_resunetpp_model.pth \
        --transfuse_ckpt transfuse.pth \
        --wdff_ckpt best_wdffnet.pth \
        --ensemble_epochs 8 \
        --output_dir /content/drive/MyDrive/ensemble_outputs/ablation
"""

import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tabulate import tabulate

from dataset import DataConfig, IMAGENET_MEAN, IMAGENET_STD, make_dataloaders
from evaluate import DEFAULT_OUTPUT_DIR, ensure_output_dir, evaluate_model, force_colab_inline, save_plot
from models import ResUNetPPWrapper, TransFuse, WDFFNet
from utils import dice_bce_loss, ensure_binary_output, measure_fps, parameter_stats, robust_load_weights, seed_everything


class AblationWeightedEnsemble(nn.Module):
    """Weighted ensemble that can fuse any subset of the available base models."""

    def __init__(self, base_models: Dict[str, nn.Module], hidden_channels: int = 16):
        super().__init__()
        if len(base_models) < 2:
            raise ValueError("Ablation ensembles require at least two base models.")

        self.model_names = list(base_models.keys())
        self.base_models = nn.ModuleDict(base_models)
        for model in self.base_models.values():
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        n_models = len(self.model_names)
        self.weight_head = nn.Sequential(
            nn.Conv2d(n_models, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, n_models, kernel_size=1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        for model in self.base_models.values():
            model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = [
                ensure_binary_output(model(x), size=x.shape[2:])
                for model in self.base_models.values()
            ]
        stack = torch.cat(logits, dim=1)
        weights = torch.softmax(self.weight_head(stack), dim=1)
        return (weights * stack).sum(dim=1, keepdim=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leave-one-model-out ablation for the weighted ensemble.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the Kvasir-SEG root containing images/ and masks/.")
    parser.add_argument("--resunet_source", type=str, default="resunet++_kvasir.py", help="Path to the ResUNet++ source file.")
    parser.add_argument("--resunet_ckpt", type=str, default="best_resunetpp_model.pth", help="Optional ResUNet++ checkpoint path.")
    parser.add_argument("--transfuse_ckpt", type=str, default="transfuse.pth", help="Optional TransFuse checkpoint path.")
    parser.add_argument("--wdff_ckpt", type=str, default="", help="Optional WDFFNet checkpoint path.")
    parser.add_argument("--image_size", type=int, default=256, help="Square image size used for dataloading and model outputs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for train/validation/test loaders.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splits and training.")
    parser.add_argument("--ensemble_epochs", type=int, default=8, help="Epochs for each fusion-head ablation run.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for each fusion head.")
    parser.add_argument("--hidden_channels", type=int, default=16, help="Hidden channels in the ensemble weighting head.")
    parser.add_argument("--fps_warmup", type=int, default=20, help="Warmup iterations for FPS measurement.")
    parser.add_argument("--fps_runs", type=int, default=100, help="Timed iterations for FPS measurement.")
    parser.add_argument("--num_samples", type=int, default=2, help="Samples to include in the side-by-side image comparison.")
    parser.add_argument("--pretrained_backbones", action="store_true", help="Download/use timm pretrained backbones before loading checkpoints.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(DEFAULT_OUTPUT_DIR, "ablation"), help="Directory for CSV, JSON, checkpoint, and PNG outputs.")
    return parser.parse_args()


def maybe_load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device, model_name: str) -> bool:
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[{model_name}] Loading checkpoint: {checkpoint_path}")
        robust_load_weights(model, checkpoint_path, device)
        return True
    print(f"[{model_name}] No checkpoint loaded from: {checkpoint_path or '<empty>'}")
    return False


def build_base_models(args: argparse.Namespace, device: torch.device) -> Tuple["OrderedDict[str, nn.Module]", Dict[str, Dict[str, object]]]:
    models: "OrderedDict[str, nn.Module]" = OrderedDict(
        [
            ("ResUNet++", ResUNetPPWrapper(repo_file=args.resunet_source, out_size=args.image_size)),
            ("TransFuse", TransFuse(out_size=args.image_size, pretrained=args.pretrained_backbones)),
            ("WDFFNet", WDFFNet(out_size=args.image_size, pretrained=args.pretrained_backbones)),
        ]
    )
    checkpoint_paths = {
        "ResUNet++": args.resunet_ckpt,
        "TransFuse": args.transfuse_ckpt,
        "WDFFNet": args.wdff_ckpt,
    }

    load_report: Dict[str, Dict[str, object]] = {}
    for name, model in models.items():
        loaded = maybe_load_checkpoint(model, checkpoint_paths[name], device, name)
        model.to(device).eval()
        total, trainable = parameter_stats(model)
        load_report[name] = {
            "checkpoint": checkpoint_paths[name],
            "checkpoint_loaded": loaded,
            "total_params": total,
            "trainable_params_before_ensemble_freeze": trainable,
        }
    return models, load_report


def build_variants(base_models: "OrderedDict[str, nn.Module]", hidden_channels: int) -> "OrderedDict[str, AblationWeightedEnsemble]":
    variants: "OrderedDict[str, AblationWeightedEnsemble]" = OrderedDict()
    variants["Full Ensemble"] = AblationWeightedEnsemble(OrderedDict(base_models), hidden_channels=hidden_channels)
    for removed_name in base_models.keys():
        kept = OrderedDict((name, model) for name, model in base_models.items() if name != removed_name)
        variants[f"Without {removed_name}"] = AblationWeightedEnsemble(kept, hidden_channels=hidden_channels)
    return variants


def train_fusion_head(
    model: AblationWeightedEnsemble,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_path: str,
) -> Tuple[Dict[str, List[float]], str]:
    model.to(device)
    optimizer = torch.optim.Adam(model.weight_head.parameters(), lr=lr)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = dice_bce_loss(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: List[float] = []
        val_dices: List[float] = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_losses.append(float(dice_bce_loss(logits, y).item()))
                probs = torch.sigmoid(ensure_binary_output(logits, size=y.shape[2:]))
                preds = (probs > 0.5).float()
                intersection = (preds * y).sum(dim=(1, 2, 3))
                denominator = preds.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
                dice = ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
                val_dices.append(float(dice.item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_dice = float(np.mean(val_dices))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        print(
            f"[{checkpoint_path}] Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history, checkpoint_path


def included_removed_names(variant_name: str, model: AblationWeightedEnsemble) -> Tuple[str, str]:
    included = ", ".join(model.model_names)
    removed = "None" if variant_name == "Full Ensemble" else variant_name.replace("Without ", "")
    return included, removed


def build_ablation_table(
    variants: "OrderedDict[str, AblationWeightedEnsemble]",
    metrics: Dict[str, Dict[str, float]],
    fps: Dict[str, float],
) -> pd.DataFrame:
    full_dice = metrics["Full Ensemble"]["Dice"]
    rows = []
    for variant_name, model in variants.items():
        total_params, trainable_params = parameter_stats(model)
        head_params = sum(param.numel() for param in model.weight_head.parameters())
        included, removed = included_removed_names(variant_name, model)
        row = {
            "Variant": variant_name,
            "Removed Model": removed,
            "Included Models": included,
            "Dice": metrics[variant_name]["Dice"],
            "IoU": metrics[variant_name]["IoU"],
            "Precision": metrics[variant_name]["Precision"],
            "Recall": metrics[variant_name]["Recall"],
            "Accuracy": metrics[variant_name]["Accuracy"],
            "Delta Dice vs Full": metrics[variant_name]["Dice"] - full_dice,
            "Total Params": total_params,
            "Trainable Params": trainable_params,
            "Fusion Head Params": head_params,
            "FPS": fps[variant_name],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_json(payload: Dict[str, object], output_dir: str, filename: str) -> str:
    path = os.path.join(ensure_output_dir(output_dir), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def plot_ablation_metrics(df: pd.DataFrame, output_dir: str) -> List[str]:
    force_colab_inline()
    saved_paths: List[str] = []

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["Variant"], df["Dice"], color=["seagreen"] + ["steelblue"] * (len(df) - 1))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice")
    ax.set_title("Ablation Study Dice: Full Ensemble vs Leave-One-Out Variants")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for patch, value in zip(ax.patches, df["Dice"]):
        ax.text(patch.get_x() + patch.get_width() / 2, min(float(value) + 0.01, 1.0), f"{value:.3f}", ha="center")
    fig.tight_layout()
    saved_paths.append(save_plot(fig, "ablation_dice_comparison.png", output_dir))
    plt.show()

    leave_one_out = df[df["Variant"] != "Full Ensemble"]
    fig_delta, ax_delta = plt.subplots(figsize=(9, 5))
    colors = ["crimson" if value < 0 else "darkorange" for value in leave_one_out["Delta Dice vs Full"]]
    ax_delta.bar(leave_one_out["Removed Model"], leave_one_out["Delta Dice vs Full"], color=colors)
    ax_delta.axhline(0, color="black", linewidth=1)
    ax_delta.set_ylabel("Dice Change Compared with Full Ensemble")
    ax_delta.set_title("Influence of Each Removed Base Model")
    ax_delta.grid(axis="y", linestyle="--", alpha=0.35)
    fig_delta.tight_layout()
    saved_paths.append(save_plot(fig_delta, "ablation_removed_model_influence.png", output_dir))
    plt.show()
    return saved_paths


def denormalize_batch(x: torch.Tensor) -> np.ndarray:
    x_np = x.detach().cpu().permute(0, 2, 3, 1).numpy()
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    return np.clip(x_np * std + mean, 0, 1)


def visualize_ablation_predictions(
    variants: "OrderedDict[str, AblationWeightedEnsemble]",
    loader,
    device: torch.device,
    num_samples: int,
    output_dir: str,
) -> str:
    force_colab_inline()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    for model in variants.values():
        model.eval()

    with torch.no_grad():
        preds = OrderedDict(
            (name, torch.sigmoid(ensure_binary_output(model(x), size=y.shape[2:])).cpu())
            for name, model in variants.items()
        )

    x_np = denormalize_batch(x)
    rows = min(num_samples, x.shape[0])
    cols = 2 + len(variants)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.asarray(axes)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx in range(rows):
        axes[row_idx, 0].imshow(x_np[row_idx])
        axes[row_idx, 0].set_title("Input")
        axes[row_idx, 1].imshow(y[row_idx, 0].detach().cpu().numpy(), cmap="gray")
        axes[row_idx, 1].set_title("Ground Truth")
        for col_idx, (variant_name, pred) in enumerate(preds.items(), start=2):
            axes[row_idx, col_idx].imshow((pred[row_idx, 0].numpy() > 0.5).astype(np.float32), cmap="gray")
            axes[row_idx, col_idx].set_title(variant_name)
        for col_idx in range(cols):
            axes[row_idx, col_idx].axis("off")

    fig.tight_layout()
    path = save_plot(fig, "ablation_prediction_comparison.png", output_dir)
    print(f"[Saved] {path}")
    plt.show()
    return path


def main() -> None:
    args = parse_args()
    force_colab_inline()
    seed_everything(args.seed)
    output_dir = ensure_output_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Output] {output_dir}")

    data_cfg = DataConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_loader, val_loader, test_loader = make_dataloaders(data_cfg)

    base_models, base_report = build_base_models(args, device)
    variants = build_variants(base_models, hidden_channels=args.hidden_channels)

    histories: Dict[str, Dict[str, List[float]]] = {}
    checkpoint_paths: Dict[str, str] = {}
    for variant_name, model in variants.items():
        safe_name = variant_name.lower().replace("++", "pp").replace(" ", "_").replace("/", "_")
        checkpoint_path = os.path.join(output_dir, f"best_{safe_name}.pth")
        history, saved_checkpoint = train_fusion_head(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.ensemble_epochs,
            lr=args.lr,
            checkpoint_path=checkpoint_path,
        )
        histories[variant_name] = history
        checkpoint_paths[variant_name] = saved_checkpoint

    metrics = {name: evaluate_model(model.to(device).eval(), test_loader, device) for name, model in variants.items()}
    fps = {
        name: measure_fps(
            model.to(device).eval(),
            device,
            input_size=(1, 3, args.image_size, args.image_size),
            warmup=args.fps_warmup,
            runs=args.fps_runs,
        )
        for name, model in variants.items()
    }
    table = build_ablation_table(variants, metrics, fps)

    table_path = os.path.join(output_dir, "ablation_results.csv")
    table.to_csv(table_path, index=False)
    print("\n=== Ablation Results ===")
    print(tabulate(table, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))
    print(f"[Saved] {table_path}")

    params_payload = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "script": "ablation_study.py",
        "experiment_parameters": vars(args),
        "device": str(device),
        "base_models": base_report,
        "ensemble_variants": {
            name: {
                "included_models": model.model_names,
                "removed_model": included_removed_names(name, model)[1],
                "checkpoint": checkpoint_paths[name],
                "history": histories[name],
                "total_params": int(parameter_stats(model)[0]),
                "trainable_params": int(parameter_stats(model)[1]),
                "fusion_head_params": int(sum(param.numel() for param in model.weight_head.parameters())),
            }
            for name, model in variants.items()
        },
        "metrics": metrics,
        "fps": fps,
    }
    parameters_path = save_json(params_payload, output_dir, "ablation_experiment_parameters.json")
    print(f"[Saved] {parameters_path}")

    plot_paths = plot_ablation_metrics(table, output_dir)
    for path in plot_paths:
        print(f"[Saved] {path}")
    comparison_path = visualize_ablation_predictions(variants, test_loader, device, args.num_samples, output_dir)
    print(f"[Saved] {comparison_path}")


if __name__ == "__main__":
    main()
