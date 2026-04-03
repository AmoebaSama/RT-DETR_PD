from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from rtdetr.inference import suggest_macro_patch_boxes


BOARD_COLOR = (109, 228, 255)
PATCH_COLOR = (70, 214, 106)
TEXT_COLOR = (16, 20, 17)
WINDOW_NAME = "RT-DETR Macro Capture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture full-scale PCB webcam frames and export macro-style solder crops for retraining."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("captured_macro_dataset"))
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "frames": output_dir / "full_frames",
        "boards": output_dir / "board_crops",
        "patches": output_dir / "macro_patches" / "unlabeled",
        "metadata": output_dir / "metadata",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def bbox_iou(first_box: list[int], second_box: list[int]) -> float:
    left = max(first_box[0], second_box[0])
    top = max(first_box[1], second_box[1])
    right = min(first_box[2], second_box[2])
    bottom = min(first_box[3], second_box[3])
    if right <= left or bottom <= top:
        return 0.0

    intersection = float((right - left) * (bottom - top))
    first_area = float(max(1, (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])))
    second_area = float(max(1, (second_box[2] - second_box[0]) * (second_box[3] - second_box[1])))
    return intersection / max(1.0, first_area + second_area - intersection)


def pil_from_bgr(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def draw_overlay(frame_bgr: np.ndarray, payload: dict, hold_remaining: float, frozen: bool) -> np.ndarray:
    canvas = frame_bgr.copy()
    board_region = payload.get("board_region")
    patch_boxes = payload.get("patch_boxes", [])

    if board_region:
        x1, y1, x2, y2 = board_region["bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOARD_COLOR, 3)
        cv2.rectangle(canvas, (x1, max(0, y1 - 34)), (x1 + 170, max(0, y1 - 2)), BOARD_COLOR, thickness=-1)
        cv2.putText(canvas, "PCB region", (x1 + 10, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)

    for patch_box in patch_boxes:
        x1, y1, x2, y2 = patch_box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), PATCH_COLOR, 2)

    status_lines = []
    if board_region:
        if frozen:
            status_lines.append("PCB locked. Press S to save, R to resume, Q to quit.")
        else:
            status_lines.append(f"PCB detected. Hold steady for {max(0.0, hold_remaining):.1f}s to lock.")
        status_lines.append(f"Macro patch proposals: {len(patch_boxes)}")
    else:
        status_lines.append("Show a full PCB to the camera. Press Q to quit.")

    line_y = 28
    for line in status_lines:
        cv2.putText(canvas, line, (18, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        line_y += 28

    return canvas


def save_capture(paths: dict[str, Path], frame_bgr: np.ndarray, payload: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    frame_path = paths["frames"] / f"{timestamp}.jpg"
    board_path = paths["boards"] / f"{timestamp}_board.jpg"
    metadata_path = paths["metadata"] / f"{timestamp}.json"

    cv2.imwrite(str(frame_path), frame_bgr)

    board_region = payload["board_region"]
    x1, y1, x2, y2 = board_region["bbox"]
    board_crop = frame_bgr[y1:y2, x1:x2]
    cv2.imwrite(str(board_path), board_crop)

    patch_files = []
    for index, patch_box in enumerate(payload.get("patch_boxes", []), start=1):
        patch_left, patch_top, patch_right, patch_bottom = patch_box
        patch_crop = frame_bgr[patch_top:patch_bottom, patch_left:patch_right]
        if patch_crop.size == 0:
            continue
        patch_path = paths["patches"] / f"{timestamp}_patch_{index:02d}.jpg"
        cv2.imwrite(str(patch_path), patch_crop)
        patch_files.append(
            {
                "path": str(patch_path),
                "bbox": patch_box,
            }
        )

    metadata = {
        "timestamp": timestamp,
        "frame_path": str(frame_path),
        "board_path": str(board_path),
        "board_region": board_region,
        "patches": patch_files,
        "image_width": payload["image_width"],
        "image_height": payload["image_height"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    args = parse_args()
    paths = ensure_dirs(args.output_dir)

    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
    camera = cv2.VideoCapture(args.camera_index, backend)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise SystemExit("Could not open the camera.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    stable_since: float | None = None
    last_board_bbox: list[int] | None = None
    frozen = False
    frozen_frame: np.ndarray | None = None
    frozen_payload: dict | None = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            if frozen and frozen_frame is not None and frozen_payload is not None:
                preview = draw_overlay(frozen_frame, frozen_payload, 0.0, frozen=True)
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("r"), ord("c")):
                    frozen = False
                    frozen_frame = None
                    frozen_payload = None
                    stable_since = None
                    last_board_bbox = None
                    continue
                if key == ord("s"):
                    metadata_path = save_capture(paths, frozen_frame, frozen_payload)
                    print(f"Saved capture set: {metadata_path}")
                    frozen = False
                    frozen_frame = None
                    frozen_payload = None
                    stable_since = None
                    last_board_bbox = None
                    continue
                continue

            ok, frame_bgr = camera.read()
            if not ok:
                continue

            frame_pil = pil_from_bgr(frame_bgr)
            payload = suggest_macro_patch_boxes(frame_pil)
            board_region = payload.get("board_region")
            now = time.monotonic()

            if board_region:
                current_bbox = board_region["bbox"]
                if last_board_bbox is None or bbox_iou(last_board_bbox, current_bbox) < 0.72:
                    stable_since = now
                elif stable_since is None:
                    stable_since = now
                last_board_bbox = current_bbox
            else:
                stable_since = None
                last_board_bbox = None

            held_seconds = 0.0 if stable_since is None else max(0.0, now - stable_since)
            preview = draw_overlay(frame_bgr, payload, args.hold_seconds - held_seconds, frozen=False)
            cv2.imshow(WINDOW_NAME, preview)

            if board_region and held_seconds >= args.hold_seconds:
                frozen = True
                frozen_frame = frame_bgr.copy()
                frozen_payload = payload
                continue

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()