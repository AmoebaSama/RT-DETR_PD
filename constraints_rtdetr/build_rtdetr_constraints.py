from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "constraints_rtdetr"
FIGURES_DIR = OUTPUT_DIR / "figures"
DATASET_ROOT = ROOT / "YOLO" / "dataset2"
RESULTS_CSV = ROOT / "rtdetr" / "runs" / "solder_defects_rtdetr" / "results.csv"
VALIDATION_ARTIFACTS_ROOT = ROOT / "runs" / "detect"

CLASS_NAMES = ["good", "exc_solder", "poor_solder", "spike"]
DISPLAY_NAMES = {
    "good": "Good Solder",
    "exc_solder": "Excess Solder",
    "poor_solder": "Insufficient Solder",
    "spike": "Solder Spike",
}
CLASS_COLORS = {
    "good": "#41b879",
    "exc_solder": "#ff8f33",
    "poor_solder": "#ff5f4d",
    "spike": "#ffcb3c",
}
VALIDATION_SUMMARY = {
    "precision": 0.9263780438404575,
    "recall": 0.9593864749751905,
    "map50": 0.9749574799277838,
    "map50_95": 0.8780956076242197,
    "class_map50_95": {
        "good": 0.93612,
        "exc_solder": 0.94187,
        "poor_solder": 0.82478,
        "spike": 0.80961,
    },
    "class_instances": {
        "good": 125,
        "exc_solder": 133,
        "poor_solder": 34,
        "spike": 101,
    },
}
LATENCY_SUMMARY = {
    "samples": 50,
    "wall_mean_ms": 418.44767400762066,
    "wall_std_ms": 40.29160827993928,
    "preprocess_mean_ms": 2.47919799759984,
    "inference_mean_ms": 409.8810420045629,
    "inference_std_ms": 13.017611691651396,
    "postprocess_mean_ms": 0.635277999099344,
    "derived_end_to_end_ms": 412.9955180012621,
}


def load_training_rows() -> list[dict[str, str]]:
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def parse_dataset_counts() -> dict[str, object]:
    split_image_counts: dict[str, int] = {}
    split_instance_counts: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}

    for split in ("train", "val"):
        label_dir = DATASET_ROOT / "labels" / split
        label_files = sorted(label_dir.glob("*.txt"))
        split_image_counts[split] = len(label_files)
        for label_path in label_files:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id = int(line.split()[0])
                split_instance_counts[split][CLASS_NAMES[class_id]] += 1

    total_counts = split_instance_counts["train"] + split_instance_counts["val"]
    smallest_class = min(CLASS_NAMES, key=lambda class_name: total_counts[class_name])
    largest_class = max(CLASS_NAMES, key=lambda class_name: total_counts[class_name])
    imbalance_ratio = total_counts[largest_class] / max(1, total_counts[smallest_class])

    return {
        "train_images": split_image_counts["train"],
        "val_images": split_image_counts["val"],
        "total_images": split_image_counts["train"] + split_image_counts["val"],
        "train_instances": dict(split_instance_counts["train"]),
        "val_instances": dict(split_instance_counts["val"]),
        "total_instances": dict(total_counts),
        "smallest_class": smallest_class,
        "smallest_class_count": int(total_counts[smallest_class]),
        "largest_class": largest_class,
        "largest_class_count": int(total_counts[largest_class]),
        "imbalance_ratio": imbalance_ratio,
        "input_resolution": "640 x 640",
        "classes": [DISPLAY_NAMES[class_name] for class_name in CLASS_NAMES],
    }


def build_summary() -> dict[str, object]:
    dataset = parse_dataset_counts()
    rows = load_training_rows()

    best_map50_row = max(rows, key=lambda row: float(row["metrics/mAP50(B)"]))
    best_map_row = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    final_row = rows[-1]

    summary = {
        "dataset": dataset,
        "performance": {
            "precision": VALIDATION_SUMMARY["precision"],
            "recall": VALIDATION_SUMMARY["recall"],
            "map50": VALIDATION_SUMMARY["map50"],
            "map50_95": VALIDATION_SUMMARY["map50_95"],
            "weakest_class": min(
                VALIDATION_SUMMARY["class_map50_95"],
                key=VALIDATION_SUMMARY["class_map50_95"].get,
            ),
            "weakest_class_map50_95": min(VALIDATION_SUMMARY["class_map50_95"].values()),
            "class_map50_95": VALIDATION_SUMMARY["class_map50_95"],
        },
        "latency": LATENCY_SUMMARY,
        "training": {
            "total_epochs": int(final_row["epoch"]),
            "final_precision": float(final_row["metrics/precision(B)"]),
            "final_recall": float(final_row["metrics/recall(B)"]),
            "final_map50": float(final_row["metrics/mAP50(B)"]),
            "final_map50_95": float(final_row["metrics/mAP50-95(B)"]),
            "best_map50_epoch": int(best_map50_row["epoch"]),
            "best_map50": float(best_map50_row["metrics/mAP50(B)"]),
            "best_map50_95_epoch": int(best_map_row["epoch"]),
            "best_map50_95": float(best_map_row["metrics/mAP50-95(B)"]),
            "training_time_seconds": float(final_row["time"]),
        },
        "data_dependency": {
            "smallest_class": dataset["smallest_class"],
            "smallest_class_count": dataset["smallest_class_count"],
            "imbalance_ratio": dataset["imbalance_ratio"],
            "weakest_class": min(
                VALIDATION_SUMMARY["class_map50_95"],
                key=VALIDATION_SUMMARY["class_map50_95"].get,
            ),
            "weakest_class_map50_95": min(VALIDATION_SUMMARY["class_map50_95"].values()),
        },
    }
    return summary


def plot_dataset_distribution(summary: dict[str, object]) -> None:
    dataset = summary["dataset"]
    train_counts = dataset["train_instances"]
    val_counts = dataset["val_instances"]
    x = range(len(CLASS_NAMES))

    plt.figure(figsize=(10, 5))
    plt.bar([value - 0.2 for value in x], [train_counts[class_name] for class_name in CLASS_NAMES], width=0.4, label="Train", color="#f68b2d")
    plt.bar([value + 0.2 for value in x], [val_counts[class_name] for class_name in CLASS_NAMES], width=0.4, label="Validation", color="#41b879")
    plt.xticks(list(x), [DISPLAY_NAMES[class_name] for class_name in CLASS_NAMES], rotation=15, ha="right")
    plt.ylabel("Object count")
    plt.title("RT-DETR dataset distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rtdetr_dataset_distribution.png", dpi=180)
    plt.close()


def plot_performance_summary(summary: dict[str, object]) -> None:
    performance = summary["performance"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    aggregate_labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
    aggregate_values = [
        performance["precision"] * 100,
        performance["recall"] * 100,
        performance["map50"] * 100,
        performance["map50_95"] * 100,
    ]
    axes[0].bar(aggregate_labels, aggregate_values, color=["#4c78a8", "#72b7b2", "#f58518", "#e45756"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Percent")
    axes[0].set_title("Validation performance summary")
    for index, value in enumerate(aggregate_values):
        axes[0].text(index, value + 1.2, f"{value:.2f}%", ha="center", fontsize=9)

    per_class = performance["class_map50_95"]
    axes[1].bar(
        [DISPLAY_NAMES[class_name] for class_name in CLASS_NAMES],
        [per_class[class_name] * 100 for class_name in CLASS_NAMES],
        color=[CLASS_COLORS[class_name] for class_name in CLASS_NAMES],
    )
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_ylabel("mAP50-95 (%)")
    axes[1].set_title("Per-class validation mAP50-95")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rtdetr_performance_summary.png", dpi=180)
    plt.close()


def plot_training_curves() -> None:
    rows = load_training_rows()
    epochs = [int(row["epoch"]) for row in rows]
    map50 = [float(row["metrics/mAP50(B)"]) * 100 for row in rows]
    map50_95 = [float(row["metrics/mAP50-95(B)"]) * 100 for row in rows]
    precision = [float(row["metrics/precision(B)"]) * 100 for row in rows]
    recall = [float(row["metrics/recall(B)"]) * 100 for row in rows]
    giou_loss = [float(row["train/giou_loss"]) for row in rows]
    cls_loss = [float(row["train/cls_loss"]) for row in rows]
    l1_loss = [float(row["train/l1_loss"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, map50, label="mAP50", color="#f58518", linewidth=2)
    axes[0].plot(epochs, map50_95, label="mAP50-95", color="#e45756", linewidth=2)
    axes[0].plot(epochs, precision, label="Precision", color="#4c78a8", linewidth=1.6, alpha=0.9)
    axes[0].plot(epochs, recall, label="Recall", color="#72b7b2", linewidth=1.6, alpha=0.9)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Percent")
    axes[0].set_title("Validation metrics across training")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(epochs, giou_loss, label="GIoU loss", color="#8c564b", linewidth=2)
    axes[1].plot(epochs, cls_loss, label="Classification loss", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, l1_loss, label="L1 loss", color="#9467bd", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training loss behavior")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rtdetr_training_curves.png", dpi=180)
    plt.close()


def plot_latency_summary(summary: dict[str, object]) -> None:
    latency = summary["latency"]
    labels = ["Preprocess", "Inference", "Postprocess", "Wall-clock"]
    values = [
        latency["preprocess_mean_ms"],
        latency["inference_mean_ms"],
        latency["postprocess_mean_ms"],
        latency["wall_mean_ms"],
    ]
    colors = ["#72b7b2", "#f58518", "#4c78a8", "#e45756"]
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values, color=colors)
    plt.ylabel("Milliseconds per image")
    plt.title("CPU latency summary for RT-DETR")
    plt.text(3, values[3] + 8, f"std {latency['wall_std_ms']:.2f} ms", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rtdetr_latency_summary.png", dpi=180)
    plt.close()


def plot_data_dependency(summary: dict[str, object]) -> None:
    dataset = summary["dataset"]
    per_class_map = summary["performance"]["class_map50_95"]
    counts = [dataset["total_instances"][class_name] for class_name in CLASS_NAMES]
    maps = [per_class_map[class_name] * 100 for class_name in CLASS_NAMES]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(
        [DISPLAY_NAMES[class_name] for class_name in CLASS_NAMES],
        counts,
        color=[CLASS_COLORS[class_name] for class_name in CLASS_NAMES],
    )
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].set_ylabel("Total labeled instances")
    axes[0].set_title("Training-data availability by class")

    for class_name, count, score in zip(CLASS_NAMES, counts, maps):
        axes[1].scatter(count, score, s=150, color=CLASS_COLORS[class_name], label=DISPLAY_NAMES[class_name])
        axes[1].text(count + 3, score + 0.6, DISPLAY_NAMES[class_name], fontsize=9)
    axes[1].set_xlabel("Total labeled instances")
    axes[1].set_ylabel("Validation mAP50-95 (%)")
    axes[1].set_title("Class frequency vs validation quality")
    axes[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rtdetr_data_dependency.png", dpi=180)
    plt.close()


def latest_validation_artifacts_dir() -> Path | None:
    candidates = []
    for path in VALIDATION_ARTIFACTS_ROOT.iterdir():
        if not path.is_dir() or not path.name.startswith("val"):
            continue
        suffix = path.name[3:]
        index = int(suffix) if suffix.isdigit() else 0
        candidates.append((index, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def copy_existing_validation_figures() -> list[str]:
    copied: list[str] = []
    latest_dir = latest_validation_artifacts_dir()
    if latest_dir is None:
        return copied
    for file_name in ["confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png"]:
        source = latest_dir / file_name
        if source.exists():
            destination = FIGURES_DIR / f"rtdetr_{file_name}"
            shutil.copy2(source, destination)
            copied.append(destination.name)
    return copied


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_summary()
    summary["copied_validation_figures"] = copy_existing_validation_figures()

    plot_dataset_distribution(summary)
    plot_performance_summary(summary)
    plot_training_curves()
    plot_latency_summary(summary)
    plot_data_dependency(summary)

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()