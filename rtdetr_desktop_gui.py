from __future__ import annotations

import argparse
import base64
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from io import BytesIO
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import cv2
import torch
from PIL import Image, ImageDraw, ImageTk

from rtdetr.inference import BOX_COLORS, detect_board_region, predict_image_bytes
from rtdetr.taxonomy import RAW_TO_DISPLAY


ROOT = Path(__file__).resolve().parent
DEFECT_LABELS = {"exc_solder", "poor_solder", "spike", "no_good"}
LEGEND_ORDER = ["good", "exc_solder", "poor_solder", "spike"]


class RTDETRDesktopGui:
    def __init__(self, root: tk.Tk, camera_index: int = 0) -> None:
        self.root = root
        self.root.title("RT-DETR Defect Inspector")
        self.root.geometry("1600x980")
        self.root.minsize(1280, 820)
        self.camera_index = camera_index
        self.camera: cv2.VideoCapture | None = None
        self.camera_job: str | None = None
        self.current_frame: Image.Image | None = None
        self.preview_image: ImageTk.PhotoImage | None = None
        self.prediction_in_flight = False
        self.last_inference_started = 0.0
        self.latest_board_region: dict | None = None
        self.freeze_after_seconds = 3.0
        self.board_visible_since: float | None = None
        self.freeze_active = False
        self.frozen_frame: Image.Image | None = None
        self.frame_interval_ms = 30
        self.inference_interval_seconds = 0.5 if torch.cuda.is_available() else 1.6

        self.configure_styles()

        container = ttk.Frame(root, padding=12)
        container.pack(fill="both", expand=True)

        controls = ttk.Frame(container)
        controls.pack(fill="x")

        ttk.Button(controls, text="Start Camera", command=self.start_camera, style="Large.TButton").pack(side="left")
        ttk.Button(controls, text="Stop Camera", command=self.stop_camera, style="Large.TButton").pack(side="left", padx=12)
        ttk.Button(controls, text="Resume Live", command=self.resume_live_feed, style="Large.TButton").pack(side="left")
        ttk.Button(controls, text="Analyze Now", command=self.run_prediction, style="Large.TButton").pack(side="left", padx=(12, 0))
        ttk.Button(controls, text="Quit", command=self.shutdown, style="Large.TButton").pack(side="right")

        device_label = "CUDA" if torch.cuda.is_available() else "CPU"
        self.status_var = tk.StringVar(
            value=f"Loaded RT-DETR live inspector | device preference: {device_label} | weights: auto"
        )
        ttk.Label(container, textvariable=self.status_var, style="Status.TLabel", wraplength=1500, justify="left").pack(
            fill="x", pady=(12, 10)
        )

        main = ttk.PanedWindow(container, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=3)
        main.add(right, weight=2)

        self.image_label = ttk.Label(left, anchor="center", text="Starting live camera feed", style="Image.TLabel")
        self.image_label.pack(fill="both", expand=True)

        self.results = ScrolledText(right, wrap="word", width=48, font=("Segoe UI", 13))
        self.results.pack(fill="both", expand=True)
        self.results.insert(
            "1.0",
            "Camera feed initializes on launch. When a PCB stays visible for 3 seconds, the frame freezes automatically for RT-DETR analysis. Use Resume Live to continue the camera feed.\n",
        )
        self.results.config(state="disabled")

        legend_frame = ttk.LabelFrame(right, text="Defect Legend", padding=8)
        legend_frame.pack(fill="x", pady=(8, 0))
        for raw_label in LEGEND_ORDER:
            row = ttk.Frame(legend_frame)
            row.pack(fill="x", pady=4)
            swatch = tk.Canvas(row, width=22, height=22, highlightthickness=0)
            color = BOX_COLORS.get(raw_label, "#ffffff")
            swatch.create_rectangle(0, 0, 22, 22, fill=color, outline="black")
            swatch.pack(side="left")
            ttk.Label(row, text=RAW_TO_DISPLAY.get(raw_label, raw_label), style="Legend.TLabel").pack(side="left", padx=10)

        self.root.after(250, self.start_camera)

    def configure_styles(self) -> None:
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=12)
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(size=12)
        heading_font = tkfont.nametofont("TkHeadingFont")
        heading_font.configure(size=13, weight="bold")

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TLabel", font=("Segoe UI", 12))
        style.configure("TLabelframe.Label", font=("Segoe UI", 13, "bold"))
        style.configure("Large.TButton", font=("Segoe UI", 12, "bold"), padding=(14, 8))
        style.configure("Status.TLabel", font=("Segoe UI", 12))
        style.configure("Legend.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Image.TLabel", font=("Segoe UI", 16), anchor="center")

    def start_camera(self) -> None:
        self.freeze_active = False
        self.frozen_frame = None
        self.board_visible_since = None
        if self.camera is not None and self.camera.isOpened():
            return
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
        capture = cv2.VideoCapture(self.camera_index, backend)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            self.status_var.set("Camera not available. Check the connected camera or camera index.")
            messagebox.showerror("Camera unavailable", "No camera feed could be opened for live inspection.")
            return
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.camera = capture
        self.status_var.set(f"Live camera feed started on camera index {self.camera_index}")
        self.schedule_camera_update()

    def stop_camera(self) -> None:
        if self.camera_job is not None:
            self.root.after_cancel(self.camera_job)
            self.camera_job = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.board_visible_since = None
        self.status_var.set("Live camera feed stopped")

    def resume_live_feed(self) -> None:
        self.freeze_active = False
        self.frozen_frame = None
        self.board_visible_since = None
        if self.camera is None or not self.camera.isOpened():
            self.start_camera()
            return
        self.status_var.set(f"Live camera feed resumed on camera index {self.camera_index}")
        if self.camera_job is None:
            self.schedule_camera_update()

    def schedule_camera_update(self) -> None:
        self.camera_job = self.root.after(self.frame_interval_ms, self.update_camera_frame)

    def update_camera_frame(self) -> None:
        self.camera_job = None
        if self.freeze_active or self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.status_var.set("Camera frame read failed. Retrying.")
            self.schedule_camera_update()
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)
        self.current_frame = pil_frame
        self.latest_board_region = detect_board_region(pil_frame)
        self.display_image(self.build_preview_frame(pil_frame, self.latest_board_region))

        now = time.monotonic()
        if self.latest_board_region is not None:
            if self.board_visible_since is None:
                self.board_visible_since = now
            visible_seconds = now - self.board_visible_since
            if visible_seconds >= self.freeze_after_seconds and not self.prediction_in_flight:
                self.freeze_frame_for_analysis(pil_frame.copy())
                return
            self.status_var.set(
                f"PCB detected. Hold steady for {max(0.0, self.freeze_after_seconds - visible_seconds):.1f}s to auto-freeze"
            )
        else:
            self.board_visible_since = None

        if not self.prediction_in_flight and now - self.last_inference_started >= self.inference_interval_seconds:
            self.launch_prediction(pil_frame.copy())

        self.schedule_camera_update()

    def freeze_frame_for_analysis(self, frame: Image.Image) -> None:
        self.freeze_active = True
        self.frozen_frame = frame
        if self.camera_job is not None:
            self.root.after_cancel(self.camera_job)
            self.camera_job = None
        self.latest_board_region = detect_board_region(frame)
        self.display_image(self.build_preview_frame(frame, self.latest_board_region))
        self.status_var.set("PCB locked. Analyzing frozen frame. Use Resume Live to continue.")
        self.launch_prediction(frame, forced=True)

    def run_prediction(self) -> None:
        if self.frozen_frame is not None:
            self.launch_prediction(self.frozen_frame.copy(), forced=True)
            return
        if self.current_frame is not None:
            self.launch_prediction(self.current_frame.copy(), forced=True)
            return
        messagebox.showwarning("No frame", "Wait for the camera feed to start before running inspection.")

    def launch_prediction(self, frame: Image.Image, forced: bool = False) -> None:
        if self.prediction_in_flight:
            return
        if not forced and time.monotonic() - self.last_inference_started < self.inference_interval_seconds:
            return
        self.prediction_in_flight = True
        self.last_inference_started = time.monotonic()
        threading.Thread(target=self.predict_on_frame, args=(frame,), daemon=True).start()

    def predict_on_frame(self, frame: Image.Image) -> None:
        try:
            buffer = BytesIO()
            frame.save(buffer, format="JPEG", quality=92)
            result = predict_image_bytes(buffer.getvalue(), imgsz=640, conf=0.5, iou=0.7)
            overlay = Image.open(BytesIO(base64.b64decode(result["annotated_image_base64"]))).convert("RGB")
            lines = self.build_result_lines(result)
            status = "PCB with solder defect" if result["overall"] == "defect" else "PCB without solder defect"
            self.root.after(0, lambda: self.apply_prediction_result(overlay, lines, result, status))
        except Exception as error:
            self.root.after(0, lambda: self.handle_prediction_error(error))

    def build_result_lines(self, result: dict) -> list[str]:
        board_region = result.get("board_region")
        class_counts = result.get("class_counts", {})
        detections = result.get("detections", [])
        defect_count = sum(
            count for label, count in class_counts.items() if label in {RAW_TO_DISPLAY.get(raw, raw) for raw in DEFECT_LABELS}
        )

        lines = [
            "Live PCB inspection",
            "",
            f"Board status: {'WITH solder defect' if result['overall'] == 'defect' else 'WITHOUT solder defect' if result['overall'] == 'good' else 'PCB detected but no solder finding'}",
            f"Inference device: {result['device_name']} ({result['device']})",
            f"Detected solder boxes: {result['total_detections']}",
            f"Frame size: {result['image_width']} x {result['image_height']}",
        ]

        if board_region:
            lines.extend(
                [
                    f"Detected PCB box: {tuple(board_region['bbox'])}",
                    f"Board coverage: {board_region['coverage'] * 100:.2f}%",
                ]
            )
        else:
            lines.append("Detected PCB box: not isolated, full frame analyzed")

        lines.extend(
            [
                "",
                f"Defect-labeled detections: {defect_count}",
                "Tip: keep the PCB centered and fill more of the frame for a tighter board crop.",
            ]
        )

        if class_counts:
            lines.extend(["", "Class summary:"])
            for label, count in class_counts.items():
                lines.append(f"- {label}: {count}")

        if detections:
            lines.extend(["", "Detection details:"])
            for index, detection in enumerate(detections, start=1):
                lines.append(
                    f"{index}. {detection['label']} | score={float(detection['confidence']):.3f} | box={tuple(detection['bbox'])}"
                )

        return lines

    def apply_prediction_result(self, overlay: Image.Image, lines: list[str], result: dict, status: str) -> None:
        self.prediction_in_flight = False
        self.display_image(overlay)
        self.results.config(state="normal")
        self.results.delete("1.0", tk.END)
        self.results.insert("1.0", "\n".join(lines))
        self.results.config(state="disabled")
        suffix = " | frozen frame" if self.freeze_active else ""
        self.status_var.set(
            f"{status} | detections={result['total_detections']} | overall={result['overall']}{suffix}"
        )

    def handle_prediction_error(self, error: Exception) -> None:
        self.prediction_in_flight = False
        self.status_var.set(f"Prediction failed: {error}")
        self.results.config(state="normal")
        self.results.delete("1.0", tk.END)
        self.results.insert("1.0", f"Prediction failed:\n{error}\n")
        self.results.config(state="disabled")

    def build_preview_frame(self, image: Image.Image, board_region: dict | None) -> Image.Image:
        preview = image.copy().convert("RGB")
        draw = ImageDraw.Draw(preview)
        outline_width = max(4, min(8, preview.width // 240))
        if board_region is not None:
            draw.rectangle(tuple(board_region["bbox"]), outline=(0, 255, 255), width=outline_width)
        return preview

    def display_image(self, image: Image.Image) -> None:
        max_width = 1050
        max_height = 860
        width, height = image.size
        scale = min(max_width / width, max_height / height, 1.0)
        resized = image.resize((int(width * scale), int(height * scale)), Image.Resampling.BILINEAR)
        self.preview_image = ImageTk.PhotoImage(resized)
        self.image_label.configure(image=self.preview_image, text="")

    def shutdown(self) -> None:
        self.stop_camera()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a desktop GUI for live RT-DETR solder defect inspection.")
    parser.add_argument("--camera-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = RTDETRDesktopGui(root, camera_index=args.camera_index)
    root.protocol("WM_DELETE_WINDOW", app.shutdown)
    root.mainloop()


if __name__ == "__main__":
    main()