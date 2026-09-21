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
            apply_contrast=False,
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
        
        # Calculate digit ratio to detect if this is a math problem
        digits = sum(c.isdigit() for c in easyocr_text)
        alpha = sum(c.isalpha() for c in easyocr_text)
        total_chars = max(1, digits + alpha)
        is_math = (digits / total_chars) > 0.3 if total_chars > 0 else False
        
        # Detect if text contains Hindi (Devanagari characters)
        # We calculate the ratio of Hindi characters to all letters. 
        # A real Hindi document will have a high ratio. A few hallucinated Hindi digits won't trigger this.
        hindi_chars = sum(1 for c in easyocr_text if '\u0900' <= c <= '\u097F')
        hindi_ratio = hindi_chars / max(1, alpha)
        is_hindi = hindi_ratio > 0.2
        print(f"DEBUG: is_math = {is_math}, hindi_chars = {hindi_chars}, alpha = {alpha}, hindi_ratio = {hindi_ratio:.2f}, is_hindi = {is_hindi}")

        # If EasyOCR confidence is low and it's not math/hindi, bypass bounding boxes and use Florence-2 VLM
        if easyocr_conf < 0.90 and not is_math and not is_hindi:
            print("DEBUG: EasyOCR confidence low. Routing full page to Florence-2 VLM.")
            florence_text, _ = recognizer.recognize_with_florence2(original_img)
            print(f"DEBUG: florence_text = {repr(florence_text)}")
            trocr_clean = postprocess_text(florence_text)

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
