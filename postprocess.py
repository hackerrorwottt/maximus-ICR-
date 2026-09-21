from spellchecker import SpellChecker
import re
import torch

def postprocess_text(text):
    """
    Basic post-processing to clean up common OCR/ICR mistakes.
    """
    if not text.strip():
        return text

    corrected_text = text
            
    # Additional common OCR cleanups can go here:
    # Remove isolated periods or commas that are often hallucinated by TrOCR crops
    corrected_text = re.sub(r'[ \t]+([.,;:?!])[ \t]+', ' ', corrected_text)
    
    # Remove excessive blank lines
    corrected_text = re.sub(r'\n{3,}', '\n\n', corrected_text)
    
    return corrected_text.strip()


