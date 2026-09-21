# Intelligent Character Recognition (ICR) Learning Project

This project demonstrates a standard computer vision pipeline for reading handwritten text, comparing traditional OCR (EasyOCR) with a modern Vision Transformer approach (TrOCR).

## Features
- **Toggleable Preprocessing**: Grayscale, contrast enhancement, noise removal, binarization, deskewing.
- **Model Comparison**: EasyOCR vs Microsoft TrOCR (Handwritten).
- **Post-processing**: Basic spell-checking and cleanup.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hackerrorwottt/maximus-ICR-.git
   cd maximus-ICR-
   ```

2. **Create and activate a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: TrOCR and PyTorch may take a while to download depending on your internet connection)*

4. **Run the web application**:
   Start the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```
   Open your browser and navigate to `http://localhost:8000` to use the web interface.

5. **Run the CLI pipeline (Optional)**:
   Alternatively, you can run the pipeline directly on an image (e.g., `sample.jpg`):
   ```bash
   python main.py sample.jpg
   ```
   You can experiment with preprocessing by toggling steps off. For example, to turn off binarization and show the preprocessing result:
   ```bash
   python main.py sample.jpg --no-bin --show-prep
   ```

## Cloud Alternative Note
If you were building this for production and didn't have privacy constraints, you might swap this local pipeline for Google Cloud Vision API (`DOCUMENT_TEXT_DETECTION`) or Azure AI Document Intelligence. They handle messy backgrounds, skewed text, and messy handwriting very well with minimal preprocessing required on your end, at the cost of requiring internet access, sending data to a third party, and API usage fees.
# maximus-ICR-
