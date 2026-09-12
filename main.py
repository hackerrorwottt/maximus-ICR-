import argparse
import cv2
import os
import warnings

# Suppress the harmless PyTorch MPS pin_memory warning on macOS
warnings.filterwarnings("ignore", message=".*'pin_memory' argument is set as true but not supported on MPS now.*")

from preprocess import preprocess_image, detect_text_lines
from recognize import ICRRecognizer
from postprocess import postprocess_text, llm_reconstruct_text

def main():
    parser = argparse.ArgumentParser(description="Intelligent Character Recognition (ICR) Pipeline")
    parser.add_argument("image_path", help="Path to the handwritten image file")
    parser.add_argument("--no-gray", action="store_true", help="Disable grayscale conversion")
    parser.add_argument('--blue-channel', action='store_true', help="Isolate the blue color channel (useful for dark text on blue backgrounds)")
    parser.add_argument("--apply-contrast", action="store_true", help="Enable contrast enhancement")
    parser.add_argument("--apply-noise", action="store_true", help="Enable noise removal")
    parser.add_argument("--apply-bin", action="store_true", help="Enable binarization")
    parser.add_argument("--no-deskew", action="store_true", help="Disable deskewing")
    parser.add_argument("--show-prep", action="store_true", help="Show original and preprocessed image")
    parser.add_argument("--log", action="store_true", help="Append results to results.csv")
    parser.add_argument("--llm-fix", action="store_true", help="Use a local LLM to logically reconstruct fragmented OCR text into perfect English.")
    
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: File not found at {args.image_path}")
        return

    print("=== Phase 1: Preprocessing ===")
    try:
        original_img, processed_img = preprocess_image(
            args.image_path,
            apply_grayscale=not args.no_gray,
            apply_blue_channel=args.blue_channel,
            apply_contrast=args.apply_contrast,
            apply_noise_removal=args.apply_noise,
            apply_binarization=args.apply_bin,
            apply_deskew=not args.no_deskew
        )
        print("Preprocessing complete.")
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return

    if args.show_prep:
        # Resize for display if the image is too large
        h, w = original_img.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            display_orig = cv2.resize(original_img, None, fx=scale, fy=scale)
            display_proc = cv2.resize(processed_img, None, fx=scale, fy=scale)
        else:
            display_orig = original_img
            display_proc = processed_img
            
        cv2.imshow("Original", display_orig)
        cv2.imshow("Preprocessed", display_proc)
        print("Press any key in the image window to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n=== Phase 2: Recognition Setup ===")
    try:
        recognizer = ICRRecognizer()
    except Exception as e:
        print(f"\n[Error] Failed to initialize recognition models: {e}")
        print("This could be due to network issues while downloading weights, missing dependencies (like PyTorch), or insufficient memory.")
        print("Please check your internet connection or ensure PyTorch is correctly installed.")
        return

    print("\n=== Phase 3: Extraction & Comparison ===")
    
    easyocr_text, easyocr_conf, easyocr_results = recognizer.recognize_with_easyocr(processed_img)
    print("\n[EasyOCR Results]")
    print(f"Raw Text:   {easyocr_text}")
    print(f"Confidence: {easyocr_conf:.2f}" if easyocr_text else "Confidence: N/A")
    
    trocr_text = ""
    # Confidence Fallback Logic
    if easyocr_conf >= 0.95:
        print("\n[Optimization] EasyOCR confidence is high (>= 0.95). Assuming clean/printed text. Skipping heavy TrOCR model to save compute.")
    else:
        print("\n[Optimization] EasyOCR confidence is low (< 0.80). Assuming messy handwriting. Running heavy TrOCR model...")
        # TrOCR Output
        trocr_text, _ = recognizer.recognize_with_trocr(processed_img, easyocr_results)
        print("\n[TrOCR Results (Cropped by EasyOCR boxes)]")
        print(f"Raw Text:   {trocr_text}")
        print("Confidence: [Not readily exposed by TrOCR API]")

        trocr_full_text, _ = recognizer.recognize_with_trocr_full_image(processed_img)
        print("\n[TrOCR Results (Full Image)]")
        print(f"Raw Text:   {trocr_full_text}")

    print("\n=== Phase 4: Post-processing ===")
    
    # Apply post-processing to both outputs for comparison
    easyocr_clean = postprocess_text(easyocr_text)
    
    print("\n[EasyOCR + Spellcheck]")
    print(easyocr_clean)
    
    if trocr_text:
        trocr_clean = postprocess_text(trocr_text)
        print("\n[TrOCR (Cropped) + Spellcheck]")
        print(trocr_clean)
        
        trocr_full_clean = postprocess_text(trocr_full_text)
        print("\n[TrOCR (Full Image) + Spellcheck]")
        print(trocr_full_clean)
    else:
        trocr_clean = "SKIPPED"
        trocr_full_clean = "SKIPPED"
        
    if args.llm_fix:
        print("\n=== Phase 5: LLM Intelligent Reconstruction ===")
        text_to_fix = trocr_clean if trocr_clean != "SKIPPED" else easyocr_clean
        llm_result = llm_reconstruct_text(text_to_fix)
        print(f"\n[Final Perfected Text]\n{llm_result}")
    
    if args.log:
        import csv
        log_file = "results.csv"
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Image", "Gray", "Contrast", "Noise", "Bin", "Deskew", "EasyOCR_Clean", "TrOCR_Clean"])
            writer.writerow([
                os.path.basename(args.image_path),
                not args.no_gray, args.apply_contrast, args.apply_noise, 
                args.apply_bin, not args.no_deskew,
                easyocr_clean, trocr_clean
            ])
        print(f"\n[Info] Results appended to {log_file}")

    print("\nPipeline Complete!")

if __name__ == "__main__":
    main()
