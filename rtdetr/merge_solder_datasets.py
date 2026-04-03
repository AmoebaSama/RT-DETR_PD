from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the original solder dataset with the converted captured dataset into one YOLO/RT-DETR dataset."
    )
    parser.add_argument("--base-dataset", type=Path, default=Path("YOLO") / "dataset2")
    parser.add_argument(
        "--captured-dataset",
        type=Path,
        default=Path("captured_macro_dataset") / "yolo_detection_dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("merged_datasets") / "solder_plus_captured",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dataset_root_and_yaml(dataset_path: Path) -> tuple[Path, Path]:
    if dataset_path.is_file():
        return dataset_path.parent, dataset_path

    candidates = [dataset_path / "data.yaml", dataset_path / "captured_macro_dataset.yaml"]
    for candidate in candidates:
        if candidate.exists():
            return dataset_path, candidate

    raise FileNotFoundError(f"Could not find a dataset YAML under {dataset_path}")


def ordered_names(names_value: dict | list) -> list[str]:
    if isinstance(names_value, list):
        return [str(item) for item in names_value]
    if isinstance(names_value, dict):
        return [str(names_value[index]) for index in sorted(int(key) for key in names_value.keys())]
    raise ValueError("Dataset names must be a list or mapping")


def ensure_clean_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "train_images": output_dir / "images" / "train",
        "val_images": output_dir / "images" / "val",
        "train_labels": output_dir / "labels" / "train",
        "val_labels": output_dir / "labels" / "val",
    }
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def copy_split(dataset_root: Path, split: str, output_image_dir: Path, output_label_dir: Path, prefix: str) -> int:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.exists() or not label_dir.exists():
        return 0

    copied = 0
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        destination_image = output_image_dir / f"{prefix}_{image_path.name}"
        shutil.copy2(image_path, destination_image)

        label_path = label_dir / f"{image_path.stem}.txt"
        destination_label = output_label_dir / f"{prefix}_{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, destination_label)
        else:
            destination_label.write_text("", encoding="utf-8")
        copied += 1
    return copied


def write_dataset_yaml(output_dir: Path, class_names: list[str]) -> Path:
    payload = {
        "path": str(output_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()
    if not args.captured_dataset.exists():
        raise SystemExit(
            "Captured dataset not found. First run the capture flow, label the patches, and build captured_macro_dataset/yolo_detection_dataset."
        )

    base_root, base_yaml_path = dataset_root_and_yaml(args.base_dataset)
    captured_root, captured_yaml_path = dataset_root_and_yaml(args.captured_dataset)
    base_yaml = load_yaml(base_yaml_path)
    captured_yaml = load_yaml(captured_yaml_path)

    base_names = ordered_names(base_yaml["names"])
    captured_names = ordered_names(captured_yaml["names"])
    if base_names != captured_names:
        raise SystemExit(
            f"Class mismatch between datasets. Base={base_names}; captured={captured_names}. They must match exactly."
        )

    output_paths = ensure_clean_dirs(args.output_dir)
    base_train = copy_split(base_root, "train", output_paths["train_images"], output_paths["train_labels"], "base")
    base_val = copy_split(base_root, "val", output_paths["val_images"], output_paths["val_labels"], "base")
    captured_train = copy_split(
        captured_root,
        "train",
        output_paths["train_images"],
        output_paths["train_labels"],
        "captured",
    )
    captured_val = copy_split(
        captured_root,
        "val",
        output_paths["val_images"],
        output_paths["val_labels"],
        "captured",
    )

    merged_yaml_path = write_dataset_yaml(args.output_dir, base_names)
    print(f"Merged dataset: {merged_yaml_path}")
    print(f"Base train images: {base_train}")
    print(f"Base val images: {base_val}")
    print(f"Captured train images: {captured_train}")
    print(f"Captured val images: {captured_val}")


if __name__ == "__main__":
    main()