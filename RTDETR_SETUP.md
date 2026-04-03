# RT-DETR Solder Defect Setup

This workspace now includes an RT-DETR training pipeline that uses your existing YOLO-format labels.

## What this model does

RT-DETR is an object detector, not a pure image classifier. In this project the current dataset supports these trained classes:

- `good` -> `Good Solder`
- `exc_solder` -> `Excess Solder`
- `poor_solder` -> `Insufficient Solder`
- `spike` -> `Solder Spike`

The current dataset does not contain labels for these requested classes:

- `Solder Bridges`
- `Misaligned Components`

Those two classes cannot be trained honestly until they are added to the annotation set.

## Additional Kaggle Dataset

An additional YOLO-format dataset can be downloaded with:

```python
import kagglehub

path = kagglehub.dataset_download("norbertelter/pcb-defect-dataset")
print(path)
```

This project now includes a helper script:

```bash
.venv\Scripts\python.exe -m rtdetr.prepare_kaggle_pcb_dataset
```

Important: the Kaggle dataset is not directly mergeable with the current solder-joint dataset.

- Current solder dataset classes: `good`, `exc_solder`, `poor_solder`, `spike`
- Kaggle dataset classes: `mouse_bite`, `spur`, `missing_hole`, `short`, `open_circuit`, `spurious_copper`

Use the Kaggle dataset as a separate full-board PCB-defect dataset or as a separate model stage. Do not merge the labels into the solder-joint RT-DETR model unless you intentionally create a new mixed taxonomy.

The default configuration uses `YOLO/dataset2`, because that dataset has the detailed defect classes.

## Offline Macro Capture Tool

To collect real full-scale webcam frames and automatically export macro-style solder crops for retraining:

```powershell
Set-Location "c:\Users\Mong\Desktop\Eden\RT-DETR"
.\.venv\Scripts\python.exe -m rtdetr.capture_macro_training_data
```

What it does:

- Opens the webcam
- Detects a PCB board in the frame
- Waits until the board stays stable for 3 seconds
- Locks the frame
- Lets you press `S` to save:
	- the full frame
	- the detected board crop
	- multiple macro-style solder patch crops
	- metadata JSON with all saved bounding boxes

Keys:

- `S`: save the locked frame and generated macro crops
- `R`: resume live capture without saving
- `Q` or `Esc`: quit

Output folders:

```text
captured_macro_dataset/
	full_frames/
	board_crops/
	macro_patches/unlabeled/
	metadata/
```

This keeps the original solder dataset untouched while creating a new dataset from your real deployment camera setup.

## Convert Captured Data Into Training Data

After capture, manually sort the saved patch images into class folders under:

```text
captured_macro_dataset/
	macro_patches/
		labeled/
			good/
			exc_solder/
			poor_solder/
			spike/
```

These folder aliases are also accepted:

- `good solder` -> `good`
- `excess solder` -> `exc_solder`
- `insufficient solder` -> `poor_solder`
- `solder spike` -> `spike`

Then build a YOLO/RT-DETR detection dataset from the captured board crops:

```powershell
Set-Location "c:\Users\Mong\Desktop\Eden\RT-DETR"
.\.venv\Scripts\python.exe -m rtdetr.prepare_captured_macro_dataset --target board
```

This writes:

```text
captured_macro_dataset/
	yolo_detection_dataset/
		images/train/
		images/val/
		labels/train/
		labels/val/
		captured_macro_dataset.yaml
		summary.json
```

Train on the converted capture dataset with:

```powershell
Set-Location "c:\Users\Mong\Desktop\Eden\RT-DETR"
.\.venv\Scripts\python.exe -m rtdetr.train --data captured_macro_dataset/yolo_detection_dataset/captured_macro_dataset.yaml --name solder_captured_rtdetr --epochs 100 --batch 4 --imgsz 640 --run-val
```

To merge the original solder dataset with the converted captured dataset into one training set:

```powershell
Set-Location "c:\Users\Mong\Desktop\Eden\RT-DETR"
.\.venv\Scripts\python.exe -m rtdetr.merge_solder_datasets
```

That writes a merged dataset to:

```text
merged_datasets/
	solder_plus_captured/
		images/train/
		images/val/
		labels/train/
		labels/val/
		data.yaml
```

Train on the merged dataset with:

```powershell
Set-Location "c:\Users\Mong\Desktop\Eden\RT-DETR"
.\.venv\Scripts\python.exe -m rtdetr.train --data merged_datasets/solder_plus_captured/data.yaml --name solder_merged_rtdetr --epochs 100 --batch 4 --imgsz 640 --run-val
```

## Environment

The local environment is:

```powershell
.\.venv\Scripts\python.exe
```

Activate it in PowerShell with:

```powershell
Set-Location "c:\Users\Mong\Desktop\Eden\RT-DETR"
.\.venv\Scripts\Activate.ps1
```

If you want the notebook to use this environment, register it as a Jupyter kernel:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name rtdetr-solder --display-name "Python (RT-DETR Solder)"
```

## Train

Train the 4-class solder defect model:

```powershell
Set-Location "c:\Users\Mong\Desktop\Eden\RT-DETR"
.\\.venv\\Scripts\\python.exe -m rtdetr.train --epochs 100 --batch 4 --imgsz 640 --run-val
```

Train the binary `good` vs `no_good` variant from `dataset1`:

```powershell
Set-Location "c:\Users\Mong\Desktop\Eden\RT-DETR"
.\\.venv\\Scripts\\python.exe -m rtdetr.train --data rtdetr/configs/solder_binary_dataset1.yaml --name solder_binary_rtdetr --epochs 100 --batch 4 --imgsz 640 --run-val
```

For a quick smoke test before a full run:

```powershell
Set-Location "c:\Users\Mong\Desktop\Eden\RT-DETR"
.\.venv\Scripts\python.exe -m rtdetr.train --epochs 1 --batch 2 --fraction 0.05 --name smoke_test --exist-ok
```

Training outputs are written to:

```text
rtdetr/runs/
```

## Validate

```powershell
Set-Location "c:\Users\Mong\Desktop\Eden\RT-DETR"
.\.venv\Scripts\python.exe -m rtdetr.validate --weights rtdetr/runs/solder_defects_rtdetr/weights/best.pt
```

## Predict

Run inference on the validation images and save annotated outputs:

```powershell
Set-Location "c:\Users\Mong\Desktop\Eden\RT-DETR"
.\.venv\Scripts\python.exe -m rtdetr.predict --weights rtdetr/runs/solder_defects_rtdetr/weights/best.pt --source YOLO/dataset2/images/val --name val_predictions
```

The prediction script prints an image-level summary such as `overall=defect; exc_solder=1` while also saving annotated images.

## GUI And API

Launch the local browser GUI:

```powershell
Set-Location "c:\Users\Mong\Desktop\Eden\RT-DETR"
.\.venv\Scripts\python.exe -m uvicorn rtdetr.web.app:app --reload
```

The deployed launcher now uses the same localhost web app entrypoint:

```powershell
.\.venv\Scripts\python.exe deployed_app\launch_rtdetr_ai.py
```

Open this in your browser:

```text
http://127.0.0.1:8000
```

The GUI now supports two input modes:

- Upload an image file
- Start the webcam and capture a frame directly in the browser

If the GPU is busy training and inference hits a CUDA memory limit, the GUI will retry inference on CPU automatically.

Postman can call the same model through a multipart form request:

- Method: `POST`
- URL: `http://127.0.0.1:8000/api/predict`
- Body type: `form-data`
- Required field: `image` as a file
- Optional fields: `imgsz`, `conf`, `iou`

The API returns JSON with:

- `overall`
- `class_counts`
- `detections`
- `annotated_image_base64`

Health check endpoint:

```text
http://127.0.0.1:8000/health
```

## Notes

- The current environment has CUDA-enabled PyTorch and can use the NVIDIA RTX 3050 Laptop GPU.
- If you run out of GPU memory, reduce `--batch` to `4` or `2`.
- `rtdetr-l.pt` is the safer default for this GPU. Use `rtdetr-x.pt` only if you want a larger model and have enough VRAM.