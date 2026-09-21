from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import shutil
import os
import cv2

from preprocess import preprocess_image
from recognize import ICRRecognizer
from postprocess import postprocess_text

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

recognizer = None

@app.on_event("startup")
def startup_event():
    global recognizer
    recognizer = ICRRecognizer()

@app.get("/")
def read_root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...)
):
    # Save the file temporarily
    file_location = f"scratch/{file.filename}"
    os.makedirs("scratch", exist_ok=True)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    try:
        # Use safe optimal settings decided by the backend (minimal interference is often best for modern OCR)
        original_img, processed_img = preprocess_image(
            file_location,
            apply_grayscale=True,
            apply_blue_channel=False,
            apply_contrast=True,
            apply_noise_removal=True,
            apply_binarization=True,
            apply_deskew=True
        )

        easyocr_text, easyocr_conf, easyocr_results = recognizer.recognize_with_easyocr(processed_img)
        
        # Heuristic: If there are many words, it's a full page document.
        # TrOCR and LLMs are excessively slow for full pages, and EasyOCR is usually sufficient.
        is_full_page = len(easyocr_results) > 20

        trocr_text = ""
        trocr_clean = "SKIPPED"
        # Only run TrOCR if confidence is low AND it's a short text/snippet.
        if easyocr_conf < 0.85 and not is_full_page:
            # Run TrOCR on the full snippet image instead of relying on EasyOCR's bounding boxes
            trocr_text, _ = recognizer.recognize_with_trocr_full_image(processed_img)
            trocr_clean = postprocess_text(trocr_text)

        easyocr_clean = postprocess_text(easyocr_text)

        final_text = trocr_clean if trocr_clean != "SKIPPED" else easyocr_clean

        if not final_text.strip():
            final_text = "NO TEXT DETECTED."

        return {"status": "success", "text": final_text, "confidence": easyocr_conf}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)
