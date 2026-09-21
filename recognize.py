import os
import torch
import cv2
import easyocr
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import sys
from unittest.mock import MagicMock
mock_flash = MagicMock()
mock_flash.__spec__ = MagicMock()
sys.modules['flash_attn'] = mock_flash

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
        self.easyocr_reader = easyocr.Reader(['en', 'hi'], gpu=use_gpu_for_easyocr)
        
        # 2. Florence-2 Setup (State-of-the-art Vision-Language Model)
        # This replaces TrOCR because it reads the entire page natively without needing bounding boxes!
        print("Initializing Florence-2 VLM...")
        self.florence_model_id = "microsoft/Florence-2-base"
        self.florence_model = AutoModelForCausalLM.from_pretrained(self.florence_model_id, trust_remote_code=True).to(self.device)
        self.florence_processor = AutoProcessor.from_pretrained(self.florence_model_id, trust_remote_code=True)
        print("Models loaded successfully.\n")

    def recognize_with_easyocr(self, image_np):
        """
        Run EasyOCR on the image numpy array.
        Returns the combined text and an average confidence score.
        """
        results = self.easyocr_reader.readtext(image_np)
        
        if not results:
            return "", 0.0, []
            
        # Filter out extreme low confidence boxes (smudges, watermarks)
        filtered_results = [r for r in results if r[2] >= 0.15]
        
        # Calculate centers
        boxes_with_centers = []
        for bbox, text, prob in filtered_results:
            tl, tr, br, bl = bbox
            cx = (tl[0] + tr[0] + br[0] + bl[0]) / 4
            cy = (tl[1] + tr[1] + br[1] + bl[1]) / 4
            h = (bl[1] + br[1]) / 2 - (tl[1] + tr[1]) / 2
            boxes_with_centers.append((bbox, text, prob, cx, cy, h))
            
        # Sort by cy first
        boxes_with_centers.sort(key=lambda x: x[4])
        
        lines = []
        current_line = []
        current_cy = None
        
        for box in boxes_with_centers:
            bbox, text, prob, cx, cy, h = box
            if current_cy is None:
                current_line.append(box)
                current_cy = cy
            else:
                # If vertical center is within half a line height, it's the same line
                if abs(cy - current_cy) < h * 0.5:
                    current_line.append(box)
                    current_cy = (current_cy * (len(current_line)-1) + cy) / len(current_line)
                else:
                    lines.append(current_line)
                    current_line = [box]
                    current_cy = cy
        if current_line:
            lines.append(current_line)
            
        # Sort each line horizontally and build text
        full_text = ""
        texts = []
        confidences = []
        detailed_results = []
        
        for i, line in enumerate(lines):
            line.sort(key=lambda x: x[3]) # Sort by cx left-to-right
            line_texts = []
            for bbox, text, prob, cx, cy, h in line:
                line_texts.append(text)
                texts.append(text)
                confidences.append(prob)
                detailed_results.append((bbox, text, prob))
            
            full_text += " ".join(line_texts)
            if i < len(lines) - 1:
                # Simple gap check to insert paragraph breaks
                next_cy = lines[i+1][0][4]
                if (next_cy - current_cy) > h * 1.5:
                    full_text += "\n\n"
                else:
                    full_text += "\n"
                current_cy = next_cy
                
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return full_text.strip(), avg_confidence, detailed_results

    def recognize_with_florence2(self, image_np):
        """
        Run Florence-2 VLM on the entire image.
        This model bypasses EasyOCR bounding boxes and natively transcribes the full page.
        """
        prompt = "<OCR>"
        
        # Convert OpenCV numpy array (BGR or Grayscale) to PIL Image (RGB)
        if len(image_np.shape) == 2: # Grayscale
            pil_image = Image.fromarray(image_np).convert("RGB")
        else: # BGR
            rgb_image = image_np[:, :, ::-1] 
            pil_image = Image.fromarray(rgb_image)

        inputs = self.florence_processor(text=prompt, images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                early_stopping=False
            )
            
        generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.florence_processor.post_process_generation(generated_text, task="<OCR>", image_size=(pil_image.width, pil_image.height))
        
        return parsed_answer["<OCR>"], None
