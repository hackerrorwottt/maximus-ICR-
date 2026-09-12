import os
import easyocr
import torch
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class ICRRecognizer:
    def __init__(self):
        """
        Initializes the OCR/ICR engines.
        We will load both EasyOCR and TrOCR to compare them.
        """
        # Automatically detect and use the best hardware accelerator (CUDA, Apple Silicon MPS, or fallback to CPU)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
            
        print(f"Hardware Acceleration: Running models on [{self.device.type.upper()}]")
        
        # 1. EasyOCR Setup
        # EasyOCR is a lightweight, widely used OCR package built on PyTorch.
        # It's generally great for printed text and standard signs, and decent for neat handwriting.
        print("Initializing EasyOCR...")
        # gpu parameter ensures it runs on the detected hardware if possible
        use_gpu_for_easyocr = self.device.type != 'cpu'
        self.easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu_for_easyocr)
        
        # 2. TrOCR Setup (Transformer-based Optical Character Recognition)
        # We check if a local fine-tuned model exists first.
        fine_tuned_path = "./fine_tuned_trocr"
        if os.path.exists(fine_tuned_path):
            print(f"Initializing custom FINE-TUNED TrOCR from {fine_tuned_path}...")
            trocr_model_id = fine_tuned_path
        else:
            print("Initializing TrOCR (base model)...")
            trocr_model_id = "microsoft/trocr-large-handwritten"
            
        self.trocr_processor = TrOCRProcessor.from_pretrained(trocr_model_id)
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_model_id).to(self.device)
        print("Models loaded successfully.\n")

    def recognize_with_easyocr(self, image_np):
        """
        Run EasyOCR on the image numpy array.
        Returns the combined text and an average confidence score.
        """
        results = self.easyocr_reader.readtext(image_np)
        
        if not results:
            return "", 0.0, []
            
        texts = []
        confidences = []
        detailed_results = []
        
        full_text = ""
        prev_y_min = None
        prev_y_max = None
        
        for (bbox, text, prob) in results:
            # Filter out extreme low confidence boxes (smudges, watermarks, paper folds)
            if prob < 0.25:
                continue
            
            texts.append(text)
            confidences.append(prob)
            detailed_results.append((bbox, text, prob))
            
            tl, tr, br, bl = bbox
            y_min = min(tl[1], tr[1])
            y_max = max(bl[1], br[1])
            
            if prev_y_max is not None:
                # If current box starts below the vertical center of the previous box, it's a new line
                prev_center = (prev_y_min + prev_y_max) / 2
                if y_min > prev_center:
                    # Calculate gap to determine if it's a new paragraph or just a new line
                    line_height = prev_y_max - prev_y_min
                    gap = y_min - prev_y_max
                    if gap > line_height * 0.5:
                        full_text += "\n\n"
                    else:
                        full_text += "\n"
                else:
                    full_text += " "
                    
            full_text += text
            prev_y_min = y_min
            prev_y_max = y_max
            
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return full_text.strip(), avg_confidence, detailed_results

    def recognize_with_trocr(self, image_np, easyocr_results):
        """
        Run TrOCR on the image numpy array.
        TrOCR expects a PIL Image.
        Returns the text. (TrOCR doesn't easily expose character-level confidence 
        in a simple API out-of-the-box like EasyOCR, so we approximate or omit it).
        
        *Learning Note*: TrOCR often performs better on contrast-enhanced grayscale 
        images rather than hard-binarized ones. If binarization was applied in preprocessing,
        you might observe worse results here compared to EasyOCR.
        
        *Crucial Note*: TrOCR is a line-level recognizer. If you feed it a full page, 
        it will hallucinate or output garbage (like "0 1"). We must use a text detector 
        (in this case, we reuse EasyOCR's bounding boxes) to crop the image into lines first.
        """
        if not easyocr_results:
            return "", None
            
        texts = []
        for bbox, easy_text, prob in easyocr_results:
            # If the EasyOCR text contains digits, or is very short (like a list bullet),
            # we trust EasyOCR because TrOCR struggles heavily with isolated numbers.
            if any(char.isdigit() for char in easy_text) or len(easy_text.strip()) <= 2:
                texts.append(easy_text)
                continue
                
            # bbox is a list of 4 points: [top-left, top-right, bottom-right, bottom-left]
            # Convert to ints and find the bounding rectangle
            tl, tr, br, bl = bbox
            x_min = max(0, int(min(tl[0], bl[0])))
            x_max = min(image_np.shape[1], int(max(tr[0], br[0])))
            y_min = max(0, int(min(tl[1], tr[1])))
            y_max = min(image_np.shape[0], int(max(bl[1], br[1])))
            
            crop_np = image_np[y_min:y_max, x_min:x_max]
            
            # Skip invalid crops
            if crop_np.size == 0 or x_max <= x_min or y_max <= y_min:
                continue

            # Convert OpenCV numpy array (BGR or Grayscale) to PIL Image (RGB)
            if len(crop_np.shape) == 2: # Grayscale
                pil_image = Image.fromarray(crop_np).convert("RGB")
            else: # BGR
                rgb_image = crop_np[:, :, ::-1] 
                pil_image = Image.fromarray(rgb_image)

            # Preprocess for the transformer model
            pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.trocr_model.generate(pixel_values, max_length=128)
                
            # Decode the tokens back to string
            generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            texts.append(generated_text)
            
        return " ".join(texts), None

    def recognize_with_trocr_full_image(self, image_np):
        """
        Run TrOCR on the entire image without using EasyOCR's bounding boxes.
        This often works much better if the image is just a single sentence or phrase,
        because EasyOCR bounding boxes can chop cursive words in half.
        """
        # Convert OpenCV numpy array (BGR or Grayscale) to PIL Image (RGB)
        if len(image_np.shape) == 2: # Grayscale
            pil_image = Image.fromarray(image_np).convert("RGB")
        else: # BGR
            rgb_image = image_np[:, :, ::-1] 
            pil_image = Image.fromarray(rgb_image)

        # Preprocess for the transformer model
        pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            # Allow longer max length since it's the whole image
            generated_ids = self.trocr_model.generate(pixel_values, max_length=256)
            
        # Decode the tokens back to string
        generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text, None
