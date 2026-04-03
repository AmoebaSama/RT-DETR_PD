from __future__ import annotations

import json
from pathlib import Path

import kagglehub
import yaml


SOLDER_TAXONOMY = [
    "good",
    "exc_solder",
    "poor_solder",
    "spike",
]


def count_files(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file())


def main() -> None:
    download_root = Path(kagglehub.dataset_download("norbertelter/pcb-defect-dataset"))
    dataset_root = download_root / "pcb-defect-dataset"
    data_yaml = yaml.safe_load((dataset_root / "data.yaml").read_text(encoding="utf-8"))

    names = data_yaml.get("names", {})
    if isinstance(names, dict):
        ordered_names = [names[index] for index in sorted(names)]
    else:
        ordered_names = list(names)

    summary = {
        "dataset_root": str(dataset_root),
        "names": ordered_names,
        "train_images": count_files(dataset_root / "train" / "images"),
        "val_images": count_files(dataset_root / "val" / "images"),
        "test_images": count_files(dataset_root / "test" / "images"),
        "compatible_with_solder_taxonomy": ordered_names == SOLDER_TAXONOMY,
        "note": (
            "This Kaggle dataset is full-board PCB manufacturing defects and is not label-compatible "
            "with the current solder-joint taxonomy. Keep it as a separate dataset or separate stage."
        ),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()