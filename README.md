# Intelligent Character Recognition (ICR) Learning Project

This project demonstrates a standard computer vision pipeline for reading handwritten text, comparing traditional OCR (EasyOCR) with a modern Vision Transformer approach (TrOCR).

## Features
- **Toggleable Preprocessing**: Grayscale, contrast enhancement, noise removal, binarization, deskewing.
- **Model Comparison**: EasyOCR vs Microsoft TrOCR (Handwritten).
- **Post-processing**: Basic spell-checking and cleanup.

## Setup Instructions

1. **Install dependencies**:
   Make sure you are in a Python virtual environment (recommended).
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: TrOCR and PyTorch may take a while to download depending on your internet connection)*

2. **Get a sample image**:
   You can take a photo of your own handwriting (write neatly on a piece of paper) and save it as `sample.jpg` in this directory. 
   Alternatively, search online for "handwritten text sample dataset" (e.g., from the IAM Handwriting Database) and download a sample image.

3. **Run the pipeline**:
   ```bash
   python main.py sample.jpg
   ```

4. **Experiment with preprocessing**:
   You can toggle steps off to see how it affects the models' ability to read the text. For example, to turn off binarization and show the preprocessing result:
   ```bash
   python main.py sample.jpg --no-bin --show-prep
   ```

## Cloud Alternative Note
If you were building this for production and didn't have privacy constraints, you might swap this local pipeline for Google Cloud Vision API (`DOCUMENT_TEXT_DETECTION`) or Azure AI Document Intelligence. They handle messy backgrounds, skewed text, and messy handwriting very well with minimal preprocessing required on your end, at the cost of requiring internet access, sending data to a third party, and API usage fees.
# maximus-ICR-
