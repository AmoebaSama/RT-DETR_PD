from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from rtdetr.taxonomy import RAW_TO_DISPLAY


RAW_CLASS_NAMES = ["good", "exc_solder", "poor_solder", "spike"]


@dataclass
class Sample:
    image_path: Path
    image_width: int
    image_height: int
    labels: list[tuple[int, list[int]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert manually labeled captured macro patches into a YOLO/RT-DETR detection dataset."
    )
    parser.add_argument("--capture-dir", type=Path, default=Path("captured_macro_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("captured_macro_dataset") / "yolo_detection_dataset")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target",
        choices=("board", "full-frame"),
        default="board",
        help="Choose whether the output images are board crops or original full frames.",
    )
    return parser.parse_args()


def normalize_label_name(label_name: str) -> str | None:
    normalized = label_name.strip().lower().replace("-", " ").replace("_", " ")
    aliases = {
        "good": "good",
        "good solder": "good",
        "exc solder": "exc_solder",
        "excess solder": "exc_solder",
        "poor solder": "poor_solder",
        "insufficient solder": "poor_solder",
        "spike": "spike",
        "solder spike": "spike",
    }
    return aliases.get(normalized)


def discover_patch_labels(labeled_root: Path) -> dict[str, str]:
    patch_labels: dict[str, str] = {}
    if not labeled_root.exists():
        return patch_labels

    for class_dir in labeled_root.iterdir():
        if not class_dir.is_dir():
            continue
        raw_label = normalize_label_name(class_dir.name)
        if raw_label is None:
            continue

        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            patch_labels[image_path.name] = raw_label
    return patch_labels


def clip_box(box: list[int], image_width: int, image_height: int) -> list[int] | None:
    left = max(0, min(image_width, box[0]))
    top = max(0, min(image_height, box[1]))
    right = max(0, min(image_width, box[2]))
    bottom = max(0, min(image_height, box[3]))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def yolo_line(class_id: int, box: list[int], image_width: int, image_height: int) -> str:
    left, top, right, bottom = box
    center_x = ((left + right) / 2.0) / image_width
    center_y = ((top + bottom) / 2.0) / image_height
    width = (right - left) / image_width
    height = (bottom - top) / image_height
    return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def load_samples(capture_dir: Path, target: str) -> list[Sample]:
    metadata_dir = capture_dir / "metadata"
    labeled_root = capture_dir / "macro_patches" / "labeled"
    patch_labels = discover_patch_labels(labeled_root)
    samples: list[Sample] = []

    for metadata_path in sorted(metadata_dir.glob("*.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        board_bbox = payload.get("board_region", {}).get("bbox")
        if not board_bbox or len(board_bbox) != 4:
            continue

        if target == "board":
            image_path = Path(payload["board_path"])
            image_width = max(1, int(board_bbox[2] - board_bbox[0]))
            image_height = max(1, int(board_bbox[3] - board_bbox[1]))
        else:
            image_path = Path(payload["frame_path"])
            image_width = int(payload["image_width"])
            image_height = int(payload["image_height"])

        if not image_path.exists():
            continue

        labels: list[tuple[int, list[int]]] = []
        for patch in payload.get("patches", []):
            raw_label = patch_labels.get(Path(patch.get("path", "")).name)
            if raw_label is None:
                continue

            box = [int(value) for value in patch["bbox"]]
            if target == "board":
                box = [
                    box[0] - int(board_bbox[0]),
                    box[1] - int(board_bbox[1]),
                    box[2] - int(board_bbox[0]),
                    box[3] - int(board_bbox[1]),
                ]

            clipped_box = clip_box(box, image_width, image_height)
            if clipped_box is None:
                continue
            labels.append((RAW_CLASS_NAMES.index(raw_label), clipped_box))

        if labels:
            samples.append(
                Sample(
                    image_path=image_path,
                    image_width=image_width,
                    image_height=image_height,
                    labels=labels,
                )
            )

    return samples


def split_samples(samples: list[Sample], val_fraction: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []

    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_count = min(val_count, len(shuffled) - 1)
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    return train_samples, val_samples


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "train_images": output_dir / "images" / "train",
        "val_images": output_dir / "images" / "val",
        "train_labels": output_dir / "labels" / "train",
        "val_labels": output_dir / "labels" / "val",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_split(samples: list[Sample], image_dir: Path, label_dir: Path) -> None:
    for sample in samples:
        destination_image = image_dir / sample.image_path.name
        shutil.copy2(sample.image_path, destination_image)
        label_path = label_dir / f"{sample.image_path.stem}.txt"
        lines = [yolo_line(class_id, box, sample.image_width, sample.image_height) for class_id, box in sample.labels]
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_yaml(output_dir: Path) -> Path:
    dataset_yaml = output_dir / "captured_macro_dataset.yaml"
    payload = {
        "path": str(output_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(RAW_CLASS_NAMES)},
    }
    dataset_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return dataset_yaml


def write_summary(output_dir: Path, train_samples: list[Sample], val_samples: list[Sample]) -> Path:
    class_counts = {name: 0 for name in RAW_CLASS_NAMES}
    for sample in [*train_samples, *val_samples]:
        for class_id, _ in sample.labels:
            class_counts[RAW_CLASS_NAMES[class_id]] += 1

    summary_path = output_dir / "summary.json"
    summary = {
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "class_counts": class_counts,
        "display_names": {name: RAW_TO_DISPLAY.get(name, name) for name in RAW_CLASS_NAMES},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    samples = load_samples(args.capture_dir, args.target)
    if not samples:
        raise SystemExit(
            "No labeled samples found. Put labeled patch images under captured_macro_dataset/macro_patches/labeled/<class>/ first."
        )

    train_samples, val_samples = split_samples(samples, args.val_fraction, args.seed)
    output_paths = ensure_output_dirs(args.output_dir)
    write_split(train_samples, output_paths["train_images"], output_paths["train_labels"])
    write_split(val_samples, output_paths["val_images"], output_paths["val_labels"])
    dataset_yaml = write_dataset_yaml(args.output_dir)
    summary_path = write_summary(args.output_dir, train_samples, val_samples)

    print(f"Prepared dataset: {dataset_yaml}")
    print(f"Summary: {summary_path}")
    print(f"Train images: {len(train_samples)}")
    print(f"Val images: {len(val_samples)}")


if __name__ == "__main__":
    main()