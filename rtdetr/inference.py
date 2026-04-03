from __future__ import annotations

import base64
from collections import Counter
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from ultralytics import RTDETR

from rtdetr.taxonomy import SUPPORTED_REQUESTED_CLASSES, UNSUPPORTED_REQUESTED_CLASSES, display_label
from rtdetr.utils import DEFAULT_WEIGHTS, RUNS_DIR, active_device_name, require_file, resolve_device


SMOKE_TEST_WEIGHTS = RUNS_DIR / "smoke_test" / "weights" / "best.pt"
GOOD_LABEL = "good"
GOOD_CONFIDENCE_FLOOR = 0.35
BOX_COLORS = {
    GOOD_LABEL: "#41b879",
    "exc_solder": "#ff8f33",
    "poor_solder": "#ff5f4d",
    "spike": "#ffcb3c",
}
BOARD_BOX_COLOR = "#6de4ff"
MIN_PATCH_SIZE = 96
MAX_PATCH_SIZE = 196
MAX_PATCH_COUNT = 28
MIN_BOARD_GUIDANCE_COVERAGE = 0.18
FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/segoeuib.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
]


def default_weights_path() -> Path:
    for candidate in (DEFAULT_WEIGHTS, SMOKE_TEST_WEIGHTS):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No trained RT-DETR weights were found. Train a model first or pass --weights explicitly."
    )


def resolve_weights_path(weights: str | Path | None = None) -> Path:
    if weights is None:
        return default_weights_path()
    return require_file(weights, "Weights")


@lru_cache(maxsize=4)
def load_model(weights_path: str) -> RTDETR:
    return RTDETR(weights_path)


def _run_prediction(
    *,
    model: RTDETR,
    image_array: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | int,
    max_det: int,
):
    return model.predict(
        source=image_array,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        max_det=max_det,
        verbose=False,
    )


def _open_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(image_bytes))
    except UnidentifiedImageError as error:
        raise ValueError("Uploaded file is not a valid image.") from error
    return image.convert("RGB")


def _encode_png_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _raw_detections(result) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or boxes.cls.numel() == 0:
        return []

    detections: list[dict[str, Any]] = []
    for class_id, confidence, xyxy in zip(boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
        detections.append(
            {
                "raw_label": result.names[class_id],
                "label": display_label(result.names[class_id]),
                "confidence": round(float(confidence), 4),
                "bbox": [round(float(value), 2) for value in xyxy],
            }
        )
    return detections


def _candidate_masks(image_bgr: np.ndarray) -> list[np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    masks: list[np.ndarray] = []

    # White PCBs are bright with relatively low saturation.
    white_mask = cv2.inRange(hsv, (0, 0, 110), (179, 85, 255))
    masks.append(white_mask)

    # Common PCB solder-mask colors.
    green_mask = cv2.inRange(hsv, (35, 30, 25), (110, 255, 255))
    blue_mask = cv2.inRange(hsv, (85, 25, 25), (140, 255, 255))
    masks.append(cv2.bitwise_or(green_mask, blue_mask))

    # Edge-based mask helps when the PCB color is not distinctive.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)
    edge_mask = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    masks.append(edge_mask)

    # Darker board-on-light-background fallback.
    _, thresh_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(thresh_inv)

    return masks


def _board_candidate_features(image_bgr: np.ndarray, bbox: list[int]) -> dict[str, float]:
    x1, y1, x2, y2 = bbox
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return {
            "white_ratio": 0.0,
            "board_color_ratio": 0.0,
            "edge_density": 0.0,
            "intensity_std": 0.0,
            "pad_blob_count": 0.0,
            "pad_blob_density": 0.0,
        }

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    white_mask = cv2.inRange(hsv, (0, 0, 110), (179, 85, 255))
    green_mask = cv2.inRange(hsv, (35, 30, 25), (110, 255, 255))
    blue_mask = cv2.inRange(hsv, (85, 25, 25), (140, 255, 255))
    board_color_mask = cv2.bitwise_or(green_mask, blue_mask)
    edges = cv2.Canny(gray, 80, 160)

    _, dark_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_inv = cv2.morphologyEx(dark_inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dark_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_area = float(max(1, roi.shape[0] * roi.shape[1]))
    pad_blob_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < roi_area * 0.00008 or area > roi_area * 0.02:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.18:
            continue
        pad_blob_count += 1

    return {
        "white_ratio": float((white_mask > 0).mean()),
        "board_color_ratio": float((board_color_mask > 0).mean()),
        "edge_density": float((edges > 0).mean()),
        "intensity_std": float(gray.std()),
        "pad_blob_count": float(pad_blob_count),
        "pad_blob_density": float(pad_blob_count / max(1.0, roi_area / 10000.0)),
    }


def _detect_board_bbox(image_array: np.ndarray) -> dict[str, Any] | None:
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    image_height, image_width = image_bgr.shape[:2]
    image_area = float(image_height * image_width)
    minimum_area = image_area * 0.05
    maximum_area = image_area * 0.65
    best_candidate: dict[str, Any] | None = None

    for raw_mask in _candidate_masks(image_bgr):
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < minimum_area or area > maximum_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) < 4 or len(approx) > 8:
                continue

            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (width, height), _ = rect
            if width <= 0 or height <= 0:
                continue

            box = cv2.boxPoints(rect).astype(np.int32)
            x, y, width_axis, height_axis = cv2.boundingRect(box)
            bbox_area = float(width * height)
            if bbox_area <= 0:
                continue

            fill_ratio = area / bbox_area
            aspect_ratio = max(width, height) / max(min(width, height), 1.0)
            if aspect_ratio < 0.7 or aspect_ratio > 2.8:
                continue
            if fill_ratio < 0.5:
                continue

            border_margin_x = image_width * 0.015
            border_margin_y = image_height * 0.015
            touches_left = x <= border_margin_x
            touches_top = y <= border_margin_y
            touches_right = (x + width_axis) >= image_width - border_margin_x
            touches_bottom = (y + height_axis) >= image_height - border_margin_y
            border_touches = sum((touches_left, touches_top, touches_right, touches_bottom))
            if border_touches >= 3:
                continue

            centeredness = 1.0 - (
                (abs(center_x - (image_width / 2)) / max(image_width / 2, 1.0)) * 0.35
                + (abs(center_y - (image_height / 2)) / max(image_height / 2, 1.0)) * 0.35
            )
            centeredness = max(centeredness, 0.1)

            coverage = round(bbox_area / image_area, 4)
            candidate_bbox = [int(x), int(y), int(x + width_axis), int(y + height_axis)]
            features = _board_candidate_features(image_bgr, candidate_bbox)
            board_surface_ratio = max(features["white_ratio"], features["board_color_ratio"])
            if board_surface_ratio < 0.18:
                continue
            if features["edge_density"] < 0.012:
                continue
            if features["intensity_std"] < 30.0:
                continue

            score = (
                area
                * fill_ratio
                * centeredness
                * max(0.5, board_surface_ratio)
                * max(0.45, features["edge_density"] * 18.0)
                * max(0.45, features["intensity_std"] / 80.0)
                / max(1, border_touches + 1)
            )
            if best_candidate is None or score > best_candidate["score"]:
                best_candidate = {
                    "bbox": candidate_bbox,
                    "score": float(score),
                    "coverage": coverage,
                    "polygon": box.reshape(-1, 2).tolist(),
                    "border_touches": int(border_touches),
                    "white_ratio": round(features["white_ratio"], 4),
                    "board_color_ratio": round(features["board_color_ratio"], 4),
                    "edge_density": round(features["edge_density"], 4),
                    "intensity_std": round(features["intensity_std"], 4),
                    "pad_blob_count": int(features["pad_blob_count"]),
                }

    return best_candidate


def detect_board_region(image: Image.Image) -> dict[str, Any] | None:
    return _detect_board_bbox(np.array(image.convert("RGB")))


def suggest_macro_patch_boxes(image: Image.Image) -> dict[str, Any]:
    image_array = np.array(image.convert("RGB"))
    image_height, image_width = image_array.shape[:2]
    board_region = _detect_board_bbox(image_array)
    if not board_region:
        return {
            "image_width": image_width,
            "image_height": image_height,
            "board_region": None,
            "patch_boxes": [],
        }

    expanded_bbox = _expand_bbox(board_region["bbox"], image_width, image_height)
    x1, y1, x2, y2 = expanded_bbox
    board_region = dict(board_region)
    board_region["bbox"] = expanded_bbox
    board_crop = image_array[y1:y2, x1:x2]

    patch_boxes = []
    for patch_left, patch_top, patch_right, patch_bottom in _generate_macro_candidate_boxes(board_crop):
        patch_boxes.append(
            [
                x1 + patch_left,
                y1 + patch_top,
                x1 + patch_right,
                y1 + patch_bottom,
            ]
        )

    return {
        "image_width": image_width,
        "image_height": image_height,
        "board_region": board_region,
        "patch_boxes": patch_boxes,
    }


def locate_board_image_bytes(image_bytes: bytes, *, include_preview: bool = False) -> dict[str, Any]:
    source_image = _open_image(image_bytes)
    board_region = detect_board_region(source_image)
    payload = {
        "image_width": source_image.width,
        "image_height": source_image.height,
        "board_detected": board_region is not None,
        "board_region": board_region,
    }
    if include_preview:
        annotated = _draw_filtered_detections(source_image, [], board_region=board_region)
        payload["annotated_image_base64"] = _encode_png_base64(annotated)
    return payload


def _expand_bbox(bbox: list[int], image_width: int, image_height: int, padding_ratio: float = 0.03) -> list[int]:
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    ]


def _offset_detections(detections: list[dict[str, Any]], offset_x: int, offset_y: int) -> list[dict[str, Any]]:
    offset_items: list[dict[str, Any]] = []
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        updated = dict(detection)
        updated["bbox"] = [
            round(x1 + offset_x, 2),
            round(y1 + offset_y, 2),
            round(x2 + offset_x, 2),
            round(y2 + offset_y, 2),
        ]
        offset_items.append(updated)
    return offset_items


def _box_iou_int(first_box: tuple[int, int, int, int], second_box: tuple[int, int, int, int]) -> float:
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


def _generate_macro_candidate_boxes(board_image: np.ndarray) -> list[tuple[int, int, int, int]]:
    board_height, board_width = board_image.shape[:2]
    if board_height == 0 or board_width == 0:
        return []

    min_dimension = min(board_width, board_height)
    patch_size = int(min(MAX_PATCH_SIZE, max(MIN_PATCH_SIZE, round(min_dimension * 0.18))))
    patch_size = min(patch_size, board_width, board_height)
    if patch_size <= 0:
        return []

    hsv = cv2.cvtColor(board_image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(board_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    bright_mask = (((hsv[:, :, 2] >= 150) & (hsv[:, :, 1] <= 110)).astype(np.uint8) * 255)
    interest = cv2.bitwise_or(bright_mask, edges)
    kernel = np.ones((3, 3), dtype=np.uint8)
    interest = cv2.dilate(interest, kernel, iterations=2)
    interest = cv2.morphologyEx(interest, cv2.MORPH_CLOSE, kernel, iterations=2)

    step = max(24, patch_size // 3)
    max_x = max(0, board_width - patch_size)
    max_y = max(0, board_height - patch_size)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for y_origin in range(0, max_y + 1, step):
        for x_origin in range(0, max_x + 1, step):
            patch_interest = interest[y_origin : y_origin + patch_size, x_origin : x_origin + patch_size]
            patch_gray = gray[y_origin : y_origin + patch_size, x_origin : x_origin + patch_size]
            if patch_interest.size == 0 or patch_gray.size == 0:
                continue
            hotspot_ratio = float((patch_interest > 0).mean())
            if hotspot_ratio < 0.035:
                continue
            sharpness = float(cv2.Laplacian(patch_gray, cv2.CV_64F).var())
            if sharpness < 18.0:
                continue
            score = hotspot_ratio + (0.0025 * sharpness)
            candidates.append((score, (x_origin, y_origin, x_origin + patch_size, y_origin + patch_size)))

    if not candidates:
        return [(0, 0, board_width, board_height)]

    candidates.sort(key=lambda item: item[0], reverse=True)
    boxes: list[tuple[int, int, int, int]] = []
    for _, candidate_box in candidates:
        if any(_box_iou_int(candidate_box, existing_box) > 0.35 for existing_box in boxes):
            continue
        boxes.append(candidate_box)
        if len(boxes) >= MAX_PATCH_COUNT:
            break

    return boxes


def _detection_iou(first_bbox: list[float], second_bbox: list[float]) -> float:
    left = max(first_bbox[0], second_bbox[0])
    top = max(first_bbox[1], second_bbox[1])
    right = min(first_bbox[2], second_bbox[2])
    bottom = min(first_bbox[3], second_bbox[3])
    if right <= left or bottom <= top:
        return 0.0

    intersection = float((right - left) * (bottom - top))
    first_area = float(max(1.0, (first_bbox[2] - first_bbox[0]) * (first_bbox[3] - first_bbox[1])))
    second_area = float(max(1.0, (second_bbox[2] - second_bbox[0]) * (second_bbox[3] - second_bbox[1])))
    return intersection / max(1.0, first_area + second_area - intersection)


def _deduplicate_detections(detections: list[dict[str, Any]], iou_threshold: float = 0.45) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        duplicate = False
        for existing in selected:
            if detection["raw_label"] != existing["raw_label"]:
                continue
            if _detection_iou(detection["bbox"], existing["bbox"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            selected.append(detection)
    return selected


def _predict_board_tiles(
    *,
    model: RTDETR,
    board_image: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | int,
    board_offset_x: int,
    board_offset_y: int,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for patch_left, patch_top, patch_right, patch_bottom in _generate_macro_candidate_boxes(board_image):
        patch = board_image[patch_top:patch_bottom, patch_left:patch_right]
        if patch.size == 0:
            continue
        patch_result = _run_prediction(
            model=model,
            image_array=patch,
            imgsz=max(imgsz, 768),
            conf=conf,
            iou=iou,
            device=device,
            max_det=120,
        )[0]
        tile_detections = _raw_detections(patch_result)
        detections.extend(
            _offset_detections(
                tile_detections,
                board_offset_x + patch_left,
                board_offset_y + patch_top,
            )
        )

    return _deduplicate_detections(detections)


def _predict_full_frame(
    *,
    model: RTDETR,
    image_array: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | int,
) -> list[dict[str, Any]]:
    results = _run_prediction(
        model=model,
        image_array=image_array,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        max_det=300,
    )
    return _raw_detections(results[0])


def _should_use_board_guidance(board_region: dict[str, Any] | None) -> bool:
    if board_region is None:
        return False
    return float(board_region.get("coverage", 0.0)) >= MIN_BOARD_GUIDANCE_COVERAGE


def _filter_detections(detections: list[dict[str, Any]], user_confidence: float) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for detection in detections:
        minimum_confidence = user_confidence
        if detection["raw_label"] == GOOD_LABEL:
            minimum_confidence = max(user_confidence, GOOD_CONFIDENCE_FLOOR)
        if detection["confidence"] >= minimum_confidence:
            filtered.append(detection)

    filtered.sort(key=lambda item: item["confidence"], reverse=True)
    return filtered


def _draw_filtered_detections(
    image: Image.Image,
    detections: list[dict[str, Any]],
    *,
    board_region: dict[str, Any] | None = None,
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    base_scale = max(image.width, image.height)
    font_size = max(20, min(36, base_scale // 24))
    line_width = max(4, font_size // 5)
    padding_x = max(10, font_size // 2)
    padding_y = max(6, font_size // 3)
    font = _load_annotation_font(font_size)

    if board_region:
        bx1, by1, bx2, by2 = board_region["bbox"]
        draw.rectangle((bx1, by1, bx2, by2), outline=BOARD_BOX_COLOR, width=max(4, line_width))
        board_text = "PCB region"
        board_box = draw.textbbox((bx1, by1), board_text, font=font)
        board_top = max(0, by1 - (board_box[3] - board_box[1]) - (padding_y * 2))
        draw.rounded_rectangle(
            (
                bx1,
                board_top,
                bx1 + (board_box[2] - board_box[0]) + (padding_x * 2),
                board_top + (board_box[3] - board_box[1]) + (padding_y * 2),
            ),
            radius=max(6, font_size // 4),
            fill=BOARD_BOX_COLOR,
        )
        draw.text((bx1 + padding_x, board_top + padding_y - 1), board_text, fill="#041118", font=font)

    for detection in detections:
        color = BOX_COLORS.get(detection["raw_label"], "#ffffff")
        x1, y1, x2, y2 = detection["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label_text = f"{detection['label']} {detection['confidence']:.2f}"
        text_box = draw.textbbox((x1, y1), label_text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        top = max(0, y1 - text_height - (padding_y * 2))
        label_box = (
            x1,
            top,
            x1 + text_width + (padding_x * 2),
            top + text_height + (padding_y * 2),
        )
        draw.rounded_rectangle(label_box, radius=max(6, font_size // 4), fill=color)
        draw.text((x1 + padding_x, top + padding_y - 1), label_text, fill="#08110d", font=font)

    return annotated


def _load_annotation_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), font_size)
            except OSError:
                continue
    return ImageFont.load_default()


def predict_image_bytes(
    image_bytes: bytes,
    *,
    weights: str | Path | None = None,
    imgsz: int = 640,
    conf: float = 0.12,
    iou: float = 0.7,
    device: str | None = None,
) -> dict[str, Any]:
    source_image = _open_image(image_bytes)
    image_array = np.array(source_image)
    image_height, image_width = image_array.shape[:2]
    weights_path = resolve_weights_path(weights)
    resolved_device = resolve_device(device)
    model = load_model(str(weights_path))
    inference_device = resolved_device
    board_region = None
    candidate_board_region = _detect_board_bbox(image_array)
    board_array = None
    offset_x = 0
    offset_y = 0

    if candidate_board_region and _should_use_board_guidance(candidate_board_region):
        expanded_bbox = _expand_bbox(candidate_board_region["bbox"], image_width, image_height)
        candidate_board_region["bbox"] = expanded_bbox
        x1, y1, x2, y2 = expanded_bbox
        board_array = image_array[y1:y2, x1:x2]
        offset_x = x1
        offset_y = y1
        board_region = candidate_board_region

    try:
        detections = _predict_full_frame(
            model=model,
            image_array=image_array,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=inference_device,
        )

        if board_region and board_array is not None:
            detections.extend(
                _predict_board_tiles(
                    model=model,
                    board_image=board_array,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    device=inference_device,
                    board_offset_x=offset_x,
                    board_offset_y=offset_y,
                )
            )
            detections.extend(
                _offset_detections(
                    _predict_full_frame(
                        model=model,
                        image_array=board_array,
                        imgsz=imgsz,
                        conf=conf,
                        iou=iou,
                        device=inference_device,
                    ),
                    offset_x,
                    offset_y,
                )
            )
    except RuntimeError as error:
        error_text = str(error).lower()
        if inference_device != "cpu" and ("out of memory" in error_text or "bad allocation" in error_text):
            inference_device = "cpu"
            detections = _predict_full_frame(
                model=model,
                image_array=image_array,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=inference_device,
            )
            if board_region and board_array is not None:
                detections.extend(
                    _predict_board_tiles(
                        model=model,
                        board_image=board_array,
                        imgsz=imgsz,
                        conf=conf,
                        iou=iou,
                        device=inference_device,
                        board_offset_x=offset_x,
                        board_offset_y=offset_y,
                    )
                )
                detections.extend(
                    _offset_detections(
                        _predict_full_frame(
                            model=model,
                            image_array=board_array,
                            imgsz=imgsz,
                            conf=conf,
                            iou=iou,
                            device=inference_device,
                        ),
                        offset_x,
                        offset_y,
                    )
                )
        else:
            raise

    detections = _filter_detections(_deduplicate_detections(detections), conf)
    class_counts: Counter[str] = Counter()

    for detection in detections:
        class_counts[detection["label"]] += 1

    overall = "no_detection"
    if detections:
        overall = "defect" if any(item["raw_label"] != GOOD_LABEL for item in detections) else "good"

    annotated_image = _draw_filtered_detections(source_image, detections, board_region=board_region)
    device_name = active_device_name() if inference_device != "cpu" else "CPU"

    return {
        "weights": str(weights_path),
        "device": str(inference_device),
        "device_name": device_name,
        "image_width": source_image.width,
        "image_height": source_image.height,
        "overall": overall,
        "total_detections": len(detections),
        "class_counts": dict(class_counts),
        "detections": detections,
        "board_region": board_region,
        "annotated_image_base64": _encode_png_base64(annotated_image),
        "supported_requested_classes": SUPPORTED_REQUESTED_CLASSES,
        "unsupported_requested_classes": UNSUPPORTED_REQUESTED_CLASSES,
    }