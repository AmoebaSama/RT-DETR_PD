from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ultralytics import RTDETR

from rtdetr.utils import DEFAULT_WEIGHTS, RUNS_DIR, active_device_name, require_file, resolve_device, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RT-DETR inference on solder defect images.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to a trained checkpoint.")
    parser.add_argument("--source", type=Path, required=True, help="Image, folder, video, or camera source.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold used for NMS.")
    parser.add_argument("--device", default=None, help="Device override, for example 0 or cpu.")
    parser.add_argument("--project", type=Path, default=RUNS_DIR / "predictions", help="Directory used for saved predictions.")
    parser.add_argument("--name", default="latest", help="Prediction run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting an existing prediction directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = require_file(args.weights, "Weights")
    source_path = require_file(args.source, "Inference source")
    project_path = resolve_path(args.project)
    device = resolve_device(args.device)

    print(f"Weights: {weights_path}")
    print(f"Source: {source_path}")
    print(f"Device: {device} ({active_device_name()})")

    model = RTDETR(str(weights_path))
    results = model.predict(
        source=str(source_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        project=str(project_path),
        name=args.name,
        exist_ok=args.exist_ok,
        save=True,
        verbose=False,
    )

    for result in results:
        boxes = result.boxes
        if boxes is None or boxes.cls.numel() == 0:
            print(f"{result.path}: no detections")
            continue

        class_ids = boxes.cls.int().tolist()
        counts = Counter(class_ids)
        class_summary = ", ".join(
            f"{result.names[class_id]}={count}" for class_id, count in sorted(counts.items())
        )
        overall = "defect" if any(class_id != 0 for class_id in class_ids) else "good"
        print(f"{result.path}: overall={overall}; {class_summary}")

    print(f"Saved predictions to: {project_path / args.name}")


if __name__ == "__main__":
    main()