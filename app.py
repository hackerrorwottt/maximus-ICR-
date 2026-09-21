from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import shutil
import os
import cv2

from preprocess import preprocess_image, detect_text_lines
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
            apply_noise_removal=False,
            apply_binarization=False,
            apply_deskew=False
        )

        easyocr_text, easyocr_conf, easyocr_results = recognizer.recognize_with_easyocr(processed_img)

        # Heuristic: If there are many words or multiple lines, it's a full document.
        print(f"DEBUG: len(easyocr_results) = {len(easyocr_results)}")
        is_full_page = len(easyocr_results) > 2 or "\n" in easyocr_text
        print(f"DEBUG: is_full_page = {is_full_page}")

        trocr_text = ""
        trocr_clean = "SKIPPED"
        
        # If EasyOCR confidence is low, leverage TrOCR (which is better at cursive/messy text)
        if easyocr_conf < 0.90:
            if not is_full_page:
                print("DEBUG: Running TrOCR full image")
                trocr_text, _ = recognizer.recognize_with_trocr_full_image(processed_img)
            else:
                print("DEBUG: Running Custom Line Detection for TrOCR")
                trocr_text, _ = recognizer.recognize_with_trocr_custom_lines(original_img, easyocr_results=easyocr_results)
                
            print(f"DEBUG: trocr_text = {repr(trocr_text)}")
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
