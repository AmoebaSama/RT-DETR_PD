from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rtdetr.inference import default_weights_path, locate_board_image_bytes, predict_image_bytes
from rtdetr.taxonomy import SUPPORTED_REQUESTED_CLASSES, UNSUPPORTED_REQUESTED_CLASSES


APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app = FastAPI(title="RT-DETR Solder Defect GUI", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


def _base_context(request: Request) -> dict[str, object]:
    static_dir = APP_DIR / "static"
    asset_version = str(
        max(
            int((static_dir / "app.js").stat().st_mtime),
            int((static_dir / "styles.css").stat().st_mtime),
        )
    )

    try:
        weights_path = default_weights_path()
        weights_display = str(weights_path)
    except FileNotFoundError:
        weights_display = "No trained weights found yet"

    return {
        "request": request,
        "result": None,
        "error": None,
        "defaults": {"imgsz": 640, "conf": 0.12, "iou": 0.7},
        "weights_display": weights_display,
        "asset_version": asset_version,
        "supported_classes": SUPPORTED_REQUESTED_CLASSES,
        "unsupported_classes": UNSUPPORTED_REQUESTED_CLASSES,
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", _base_context(request))


@app.get("/health")
async def health() -> JSONResponse:
    try:
        weights = str(default_weights_path())
    except FileNotFoundError:
        weights = None
    return JSONResponse({"status": "ok", "weights": weights})


@app.post("/", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    image: UploadFile = File(...),
    imgsz: int = Form(640),
    conf: float = Form(0.12),
    iou: float = Form(0.7),
) -> HTMLResponse:
    context = _base_context(request)
    context["defaults"] = {"imgsz": imgsz, "conf": conf, "iou": iou}

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise ValueError("Please upload an image file.")
        context["result"] = predict_image_bytes(image_bytes, imgsz=imgsz, conf=conf, iou=iou)
    except Exception as error:
        context["error"] = str(error)

    return templates.TemplateResponse(request, "index.html", context)


@app.post("/api/predict")
async def predict_api(
    image: UploadFile = File(...),
    imgsz: int = Form(640),
    conf: float = Form(0.12),
    iou: float = Form(0.7),
) -> JSONResponse:
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise ValueError("Please upload an image file.")
        result = predict_image_bytes(image_bytes, imgsz=imgsz, conf=conf, iou=iou)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return JSONResponse(result)


@app.post("/api/locate-board")
async def locate_board_api(image: UploadFile = File(...)) -> JSONResponse:
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise ValueError("Please upload an image file.")
        result = locate_board_image_bytes(image_bytes, include_preview=True)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return JSONResponse(result)