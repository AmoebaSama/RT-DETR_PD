from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR

from rtdetr.utils import DEFAULT_DATASET, DEFAULT_WEIGHTS, active_device_name, prepare_dataset_config, require_file, resolve_device, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained RT-DETR solder defect model.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to a trained checkpoint.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET, help="Dataset YAML file.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate.")
    parser.add_argument("--device", default=None, help="Device override, for example 0 or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = require_file(args.weights, "Weights")
    source_data_path = resolve_path(args.data)
    data_path = prepare_dataset_config(args.data)
    device = resolve_device(args.device)

    print(f"Weights: {weights_path}")
    print(f"Dataset: {source_data_path}")
    print(f"Resolved dataset: {data_path}")
    print(f"Device: {device} ({active_device_name()})")

    model = RTDETR(str(weights_path))
    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        device=device,
        plots=True,
    )

    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()