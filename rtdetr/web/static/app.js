(function () {
    const imgszInput = document.getElementById("imgsz-input");
    const confInput = document.getElementById("conf-input");
    const iouInput = document.getElementById("iou-input");
    const errorBanner = document.getElementById("error-banner");
    const errorText = document.getElementById("error-text");
    const webcamVideo = document.getElementById("webcam-video");
    const webcamCanvas = document.getElementById("webcam-canvas");
    const liveOverlay = document.getElementById("live-overlay");
    const analysisImage = document.getElementById("analysis-image");
    const previewBadge = document.getElementById("preview-badge");
    const stagePlaceholder = document.getElementById("stage-placeholder");
    const statusChip = document.getElementById("status-chip");
    const metaBlock = document.getElementById("meta-block");
    const boardRegionContent = document.getElementById("board-region-content");
    const classCountsContent = document.getElementById("class-counts-content");
    const detectionDetailsContent = document.getElementById("detection-details-content");
    const uploadInput = document.getElementById("upload-input");
    const analyzeUploadButton = document.getElementById("analyze-upload");
    const clearUploadButton = document.getElementById("clear-upload");
    const uploadStatus = document.getElementById("upload-status");
    const startCameraButton = document.getElementById("start-camera");
    const startLiveButton = document.getElementById("start-live");
    const pauseLiveButton = document.getElementById("pause-live");
    const snapshotFrameButton = document.getElementById("snapshot-frame");
    const stopCameraButton = document.getElementById("stop-camera");
    const liveStatus = document.getElementById("live-status");

    let webcamStream = null;
    let liveLoopId = null;
    let isSendingFrame = false;
    let stableBoardSince = null;
    let lastBoardRegion = null;
    let uploadPreviewUrl = null;

    const STABLE_BOARD_MS = 3000;
    const BOARD_SCAN_INTERVAL_MS = 800;

    function bboxIoU(boxA, boxB) {
        if (!boxA || !boxB) {
            return 0;
        }

        const left = Math.max(boxA[0], boxB[0]);
        const top = Math.max(boxA[1], boxB[1]);
        const right = Math.min(boxA[2], boxB[2]);
        const bottom = Math.min(boxA[3], boxB[3]);
        if (right <= left || bottom <= top) {
            return 0;
        }

        const intersection = (right - left) * (bottom - top);
        const areaA = Math.max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]));
        const areaB = Math.max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]));
        return intersection / (areaA + areaB - intersection);
    }

    function escapeHtml(value) {
        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }

    function currentSettings() {
        return {
            imgsz: imgszInput.value || "640",
            conf: confInput.value || "0.12",
            iou: iouInput.value || "0.7",
        };
    }

    function setLiveStatus(message) {
        liveStatus.textContent = message;
    }

    function setUploadStatus(message) {
        uploadStatus.textContent = message;
    }

    function showError(message) {
        errorText.textContent = message;
        errorBanner.classList.remove("hidden");
    }

    function clearError() {
        errorText.textContent = "";
        errorBanner.classList.add("hidden");
    }

    function setStatus(kind, label, metaItems) {
        statusChip.className = `status-chip ${kind}`;
        statusChip.textContent = label;
        metaBlock.innerHTML = (metaItems || []).map((item) => `<span>${escapeHtml(item)}</span>`).join("");
    }

    function setBoardRegionContent(html) {
        boardRegionContent.innerHTML = html;
    }

    function setClassCountsContent(html) {
        classCountsContent.innerHTML = html;
    }

    function setDetectionDetailsContent(html) {
        detectionDetailsContent.innerHTML = html;
    }

    function showLiveStage() {
        webcamVideo.classList.remove("hidden");
        liveOverlay.classList.remove("hidden");
        analysisImage.classList.add("hidden");
        previewBadge.classList.add("hidden");
        stagePlaceholder.classList.add("hidden");
    }

    function showPlaceholder(message) {
        webcamVideo.classList.add("hidden");
        liveOverlay.classList.add("hidden");
        analysisImage.classList.add("hidden");
        previewBadge.classList.add("hidden");
        stagePlaceholder.textContent = message;
        stagePlaceholder.classList.remove("hidden");
    }

    function showUploadPreviewStage(objectUrl, fileName) {
        analysisImage.src = objectUrl;
        analysisImage.classList.remove("hidden");
        webcamVideo.classList.add("hidden");
        liveOverlay.classList.add("hidden");
        previewBadge.textContent = `Selected upload: ${fileName}`;
        previewBadge.classList.remove("hidden");
        stagePlaceholder.classList.add("hidden");
    }

    function showAnalysisStage(imageBase64) {
        analysisImage.src = `data:image/png;base64,${imageBase64}`;
        analysisImage.classList.remove("hidden");
        webcamVideo.classList.add("hidden");
        liveOverlay.classList.add("hidden");
        previewBadge.classList.add("hidden");
        stagePlaceholder.classList.add("hidden");
    }

    function clearOverlay() {
        const context = liveOverlay.getContext("2d");
        context.clearRect(0, 0, liveOverlay.width, liveOverlay.height);
    }

    function drawBoardOverlay(boardRegion) {
        const videoWidth = webcamVideo.videoWidth;
        const videoHeight = webcamVideo.videoHeight;
        if (!videoWidth || !videoHeight) {
            return;
        }

        const displayWidth = webcamVideo.clientWidth;
        const displayHeight = webcamVideo.clientHeight;
        liveOverlay.width = displayWidth;
        liveOverlay.height = displayHeight;
        const context = liveOverlay.getContext("2d");
        context.clearRect(0, 0, displayWidth, displayHeight);

        if (!boardRegion) {
            return;
        }

        const [x1, y1, x2, y2] = boardRegion.bbox;
        const scaleX = displayWidth / videoWidth;
        const scaleY = displayHeight / videoHeight;
        const left = x1 * scaleX;
        const top = y1 * scaleY;
        const width = (x2 - x1) * scaleX;
        const height = (y2 - y1) * scaleY;

        context.strokeStyle = "#6de4ff";
        context.lineWidth = 4;
        context.strokeRect(left, top, width, height);
        context.fillStyle = "#6de4ff";
        context.fillRect(left, Math.max(0, top - 34), 140, 34);
        context.fillStyle = "#041118";
        context.font = "bold 22px Bahnschrift";
        context.fillText("PCB region", left + 10, Math.max(24, top - 10));
    }

    function resetBoardTracking() {
        stableBoardSince = null;
        lastBoardRegion = null;
        clearOverlay();
    }

    function renderResult(result) {
        const countEntries = Object.entries(result.class_counts || {});
        const detectionEntries = result.detections || [];
        const boardRegion = result.board_region || null;
        const countsMarkup = countEntries.length
            ? countEntries
                .map(([label, count]) => `<li><span>${escapeHtml(label)}</span><strong>${count}</strong></li>`)
                .join("")
            : '<p class="muted">No detections were returned for this image.</p>';

        const detectionsMarkup = detectionEntries.length
            ? detectionEntries
                .map((item) => `
                    <li>
                        <div>
                            <strong>${escapeHtml(item.label)}</strong>
                            <span class="mono">${escapeHtml(JSON.stringify(item.bbox))}</span>
                        </div>
                        <span>${(item.confidence * 100).toFixed(2)}%</span>
                    </li>
                `)
                .join("")
            : '<p class="muted">Nothing crossed the current confidence threshold.</p>';

        const boardMarkup = boardRegion
            ? `
                <ul class="summary-list">
                    <li><span>Bounding Box</span><strong>${escapeHtml(JSON.stringify(boardRegion.bbox))}</strong></li>
                    <li><span>Frame Coverage</span><strong>${(boardRegion.coverage * 100).toFixed(2)}%</strong></li>
                </ul>
            `
            : '<p class="muted">No clear board boundary was isolated, so the full frame was analyzed.</p>';

        setStatus(
            result.overall,
            result.overall.replaceAll("_", " "),
            [
                `${result.total_detections} detections`,
                `${result.image_width} x ${result.image_height}`,
                `${result.device_name} via ${result.device}`,
                boardRegion ? "Board localized" : "Board fallback: full frame",
            ]
        );
        setBoardRegionContent(boardMarkup);
        setClassCountsContent(countEntries.length ? `<ul class="summary-list">${countsMarkup}</ul>` : countsMarkup);
        setDetectionDetailsContent(detectionEntries.length ? `<ul class="detection-list">${detectionsMarkup}</ul>` : detectionsMarkup);
        showAnalysisStage(result.annotated_image_base64);
    }

    function renderBoardScan(result, secondsHeld) {
        const boardRegion = result.board_region || null;
        const boardMarkup = boardRegion
            ? `
                <ul class="summary-list">
                    <li><span>Bounding Box</span><strong>${escapeHtml(JSON.stringify(boardRegion.bbox))}</strong></li>
                    <li><span>Frame Coverage</span><strong>${(boardRegion.coverage * 100).toFixed(2)}%</strong></li>
                    <li><span>Hold Time</span><strong>${secondsHeld.toFixed(1)} s / 3.0 s</strong></li>
                </ul>
            `
            : '<p class="muted">No PCB has been isolated yet. Center the board and reduce background clutter.</p>';

        setStatus(boardRegion ? "good" : "no_detection", boardRegion ? "board detected" : "searching", [
            `${result.image_width} x ${result.image_height}`,
            boardRegion ? "PCB localized" : "Waiting for PCB",
            `${secondsHeld.toFixed(1)} s hold`,
        ]);
        setBoardRegionContent(boardMarkup);
        setClassCountsContent('<p class="muted">RT-DETR will run after the PCB stays stable for 3 seconds.</p>');
        setDetectionDetailsContent('<p class="muted">No defect analysis yet. The app is only tracking the PCB boundary.</p>');
    }

    async function captureCurrentFrameBlob() {
        if (!webcamStream || !webcamVideo.videoWidth || !webcamVideo.videoHeight) {
            throw new Error("Camera is not ready yet.");
        }
        webcamCanvas.width = webcamVideo.videoWidth;
        webcamCanvas.height = webcamVideo.videoHeight;
        const context = webcamCanvas.getContext("2d");
        context.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

        const blob = await new Promise((resolve) => webcamCanvas.toBlob(resolve, "image/jpeg", 0.92));
        if (!blob) {
            throw new Error("Could not capture a frame from the webcam.");
        }
        return blob;
    }

    async function postFrame(endpoint, blob) {
        const settings = currentSettings();
        const formData = new FormData();
        formData.append("image", blob, "live-frame.jpg");
        if (endpoint === "/api/predict") {
            formData.append("imgsz", settings.imgsz);
            formData.append("conf", settings.conf);
            formData.append("iou", settings.iou);
        }

        const response = await fetch(endpoint, {
            method: "POST",
            body: formData,
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || "Request failed.");
        }
        return payload;
    }

    async function locateBoardFromCurrentFrame() {
        if (isSendingFrame) {
            return null;
        }

        isSendingFrame = true;
        try {
            const blob = await captureCurrentFrameBlob();
            return await postFrame("/api/locate-board", blob);
        } finally {
            isSendingFrame = false;
        }
    }

    async function sendPrediction(formData) {
        clearError();

        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });

        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || "Prediction request failed.");
        }

        renderResult(payload);
        return payload;
    }

    async function analyzeUploadedFile() {
        const file = uploadInput.files && uploadInput.files[0];
        if (!file) {
            showError("Choose an image file first.");
            return;
        }

        stopLiveLoop();
        clearError();
        setUploadStatus(`Analyzing ${file.name}...`);
        setStatus("good", "analyzing", [file.name, `${Math.round(file.size / 1024)} KB`]);

        const settings = currentSettings();
        const formData = new FormData();
        formData.append("image", file, file.name);
        formData.append("imgsz", settings.imgsz);
        formData.append("conf", settings.conf);
        formData.append("iou", settings.iou);

        const payload = await sendPrediction(formData);
        setUploadStatus(`Analyzed ${file.name}`);
        return payload;
    }

    async function analyzeCurrentFrame(loadingMessage) {
        if (isSendingFrame) {
            return null;
        }

        isSendingFrame = true;
        try {
            setStatus("good", "analyzing", [loadingMessage]);
            const blob = await captureCurrentFrameBlob();
            const settings = currentSettings();
            const formData = new FormData();
            formData.append("image", blob, "live-frame.jpg");
            formData.append("imgsz", settings.imgsz);
            formData.append("conf", settings.conf);
            formData.append("iou", settings.iou);
            return await sendPrediction(formData);
        } finally {
            isSendingFrame = false;
        }
    }

    function stopLiveLoop() {
        if (liveLoopId !== null) {
            window.clearInterval(liveLoopId);
            liveLoopId = null;
        }
        resetBoardTracking();
        startLiveButton.disabled = !webcamStream;
        pauseLiveButton.disabled = true;
    }

    async function runBoardScanPass() {
        try {
            const payload = await locateBoardFromCurrentFrame();
            if (!payload) {
                return;
            }

            const boardRegion = payload.board_region;
            showLiveStage();
            drawBoardOverlay(boardRegion);
            const now = Date.now();
            if (boardRegion) {
                const isStable = lastBoardRegion && bboxIoU(lastBoardRegion.bbox, boardRegion.bbox) >= 0.7;
                if (!stableBoardSince || !isStable) {
                    stableBoardSince = now;
                }
                lastBoardRegion = boardRegion;
                const heldForMs = now - stableBoardSince;
                renderBoardScan(payload, heldForMs / 1000);

                if (heldForMs >= STABLE_BOARD_MS) {
                    stopLiveLoop();
                    setLiveStatus("PCB held steady for 3 seconds. Running RT-DETR analysis on frozen frame...");
                    await analyzeCurrentFrame("PCB locked. Running RT-DETR analysis...");
                    setLiveStatus("PCB locked and analyzed. Click Resume Live or Start Camera to continue.");
                    return;
                }

                const remaining = Math.max(0, (STABLE_BOARD_MS - heldForMs) / 1000);
                setLiveStatus(`PCB detected. Hold steady for ${remaining.toFixed(1)}s to auto-analyze.`);
            } else {
                resetBoardTracking();
                renderBoardScan(payload, 0);
                setLiveStatus("Searching for PCB board. Center it and reduce background clutter.");
            }
        } catch (error) {
            stopLiveLoop();
            showError(error.message || "Prediction failed.");
            setLiveStatus("Board scan paused after an error.");
        }
    }

    startCameraButton.addEventListener("click", async () => {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment" },
                audio: false,
            });
            webcamVideo.srcObject = webcamStream;
            await webcamVideo.play();
            showLiveStage();
            startLiveButton.disabled = false;
            snapshotFrameButton.disabled = false;
            stopCameraButton.disabled = false;
            startCameraButton.disabled = true;
            setStatus("no_detection", "camera ready", ["Waiting for PCB board"]);
            setLiveStatus("Camera ready. Start board scan when the PCB is in frame.");
            clearError();
        } catch (error) {
            showError("Could not access the camera. Check browser permissions and use localhost.");
        }
    });

    stopCameraButton.addEventListener("click", () => {
        stopLiveLoop();
        if (webcamStream) {
            webcamStream.getTracks().forEach((track) => track.stop());
            webcamStream = null;
        }
        webcamVideo.srcObject = null;
        snapshotFrameButton.disabled = true;
        stopCameraButton.disabled = true;
        startCameraButton.disabled = false;
        startLiveButton.disabled = true;
        pauseLiveButton.disabled = true;
        showPlaceholder("Start the camera and begin live analysis to show the webcam and bounding boxes here.");
        setStatus("no_detection", "idle", ["Waiting for camera"]);
        setLiveStatus("Camera idle");
    });

    startLiveButton.addEventListener("click", async () => {
        if (!webcamStream) {
            showError("Start the camera first.");
            return;
        }
        startLiveButton.disabled = true;
        pauseLiveButton.disabled = false;
        resetBoardTracking();
        await webcamVideo.play();
        showLiveStage();
        setLiveStatus("Board scan running...");
        await runBoardScanPass();
        if (webcamStream) {
            liveLoopId = window.setInterval(runBoardScanPass, BOARD_SCAN_INTERVAL_MS);
        }
    });

    pauseLiveButton.addEventListener("click", () => {
        stopLiveLoop();
        clearOverlay();
        setLiveStatus("Board scan paused.");
    });

    analyzeUploadButton.addEventListener("click", async () => {
        try {
            await analyzeUploadedFile();
        } catch (error) {
            showError(error.message || "Prediction failed.");
            setUploadStatus("Upload analysis failed.");
        }
    });

    clearUploadButton.addEventListener("click", () => {
        uploadInput.value = "";
        if (uploadPreviewUrl) {
            URL.revokeObjectURL(uploadPreviewUrl);
            uploadPreviewUrl = null;
        }
        setUploadStatus("No file selected");
        showPlaceholder("Choose an upload or start the camera to show an image here.");
    });

    uploadInput.addEventListener("change", () => {
        const file = uploadInput.files && uploadInput.files[0];
        if (uploadPreviewUrl) {
            URL.revokeObjectURL(uploadPreviewUrl);
            uploadPreviewUrl = null;
        }

        if (!file) {
            setUploadStatus("No file selected");
            showPlaceholder("Choose an upload or start the camera to show an image here.");
            return;
        }

        uploadPreviewUrl = URL.createObjectURL(file);
        setUploadStatus(`Selected ${file.name}`);
        setStatus("no_detection", "upload ready", [file.name, `${Math.round(file.size / 1024)} KB`]);
        setBoardRegionContent('<p class="muted">Preview loaded. Click Analyze Upload to run RT-DETR on this image.</p>');
        setClassCountsContent('<p class="muted">No detections yet. Run upload analysis to populate results.</p>');
        setDetectionDetailsContent('<p class="muted">Detection details will appear after upload analysis finishes.</p>');
        showUploadPreviewStage(uploadPreviewUrl, file.name);
    });

    snapshotFrameButton.addEventListener("click", async () => {
        try {
            stopLiveLoop();
            showLiveStage();
            setLiveStatus("Analyzing current frozen frame...");
            const payload = await analyzeCurrentFrame("Analyzing one live frame...");
            setLiveStatus(payload && payload.board_region ? "Frozen frame analyzed." : "Frozen frame analyzed using full-frame fallback.");
        } catch (error) {
            showError(error.message || "Prediction failed.");
        }
    });

    document.addEventListener("visibilitychange", () => {
        if (document.hidden && liveLoopId !== null) {
            stopLiveLoop();
            setLiveStatus("Board scan paused because the page is not visible.");
        }
    });

    showPlaceholder("Choose an upload or start the camera to show an image here.");
    setUploadStatus("No file selected");
})();