from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR

from rtdetr.utils import DEFAULT_DATASET, RUNS_DIR, active_device_name, prepare_dataset_config, resolve_device, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RT-DETR model for solder defect detection.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET, help="Dataset YAML file.")
    parser.add_argument("--model", default="rtdetr-l.pt", help="Pretrained RT-DETR checkpoint to fine-tune.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--optimizer", default="auto", help="Optimizer name passed to Ultralytics.")
    parser.add_argument("--device", default=None, help="Device override, for example 0 or cpu.")
    parser.add_argument("--name", default="solder_defects_rtdetr", help="Run name under rtdetr/runs.")
    parser.add_argument("--project", type=Path, default=RUNS_DIR, help="Directory used for training outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cache", action="store_true", help="Cache images in memory for faster training.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting an existing run directory.")
    parser.add_argument("--run-val", action="store_true", help="Run validation after training finishes.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the dataset to use.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_data_path = resolve_path(args.data)
    data_path = prepare_dataset_config(args.data)
    project_path = resolve_path(args.project)
    device = resolve_device(args.device)

    print(f"Dataset: {source_data_path}")
    print(f"Resolved dataset: {data_path}")
    print(f"Model: {args.model}")
    print(f"Device: {device} ({active_device_name()})")
    print(f"Outputs: {project_path / args.name}")

    model = RTDETR(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        device=device,
        project=str(project_path),
        name=args.name,
        seed=args.seed,
        fraction=args.fraction,
        cache=args.cache,
        exist_ok=args.exist_ok,
        pretrained=True,
        amp=True,
        plots=True,
        val=True,
    )

    best_weights = project_path / args.name / "weights" / "best.pt"
    print(f"Best checkpoint: {best_weights}")

    if args.run_val and best_weights.exists():
        print("Running validation on the best checkpoint...")
        validated_model = RTDETR(str(best_weights))
        validated_model.val(data=str(data_path), imgsz=args.imgsz, batch=args.batch, device=device)


if __name__ == "__main__":
    main()