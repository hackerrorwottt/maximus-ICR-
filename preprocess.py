import cv2
import numpy as np

def preprocess_image(image_path, 
                     apply_grayscale=True, 
                     apply_blue_channel=False,
                     apply_contrast=False, 
                     apply_noise_removal=False, 
                     apply_binarization=False, 
                     apply_deskew=True):
    """
    Preprocess an image for ICR.
    Each step is toggleable to understand its effect on the output.
    """
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    # Note on Cloud APIs (Google Vision / Azure Form Recognizer):
    # Cloud Vision APIs often have very robust, built-in preprocessing. 
    # If using a cloud API, much of this pipeline could be skipped or minimized 
    # because they handle skew, uneven lighting, and noise automatically. 
    # However, running locally gives us full control and privacy.

    processed = img.copy()

    # 2. Color Channel Processing (Grayscale or Blue Channel Isolation)
    # Why it matters for ICR: Handwriting can be in various ink colors (blue, black, red) 
    # and lighting can create false color edges. Grayscale or channel isolation reduces the 
    # data from 3 channels (RGB) to 1 channel (intensity), simplifying the image and 
    # reducing computational load while preserving the structural information of the text.
    if apply_blue_channel:
        # BGR format, index 0 is Blue. 
        # A blue background becomes bright, and black ink stays dark, maximizing contrast.
        processed = processed[:, :, 0]
    elif apply_grayscale and len(processed.shape) == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # 3. Contrast Enhancement (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    # Why it matters for ICR: Photos of handwritten notes often suffer from uneven lighting or shadows. 
    # Faded ink can also blend into the page. CLAHE locally enhances the contrast in small 
    # tiles of the image, making the strokes stand out clearly against the background.
    if apply_contrast:
        if len(processed.shape) == 3:
             processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        processed = clahe.apply(processed)

    # 4. Noise Removal (Blurring)
    # Why it matters for ICR: Paper texture, dust, or sensor noise can create tiny dots that 
    # the OCR engine might misinterpret as punctuation (like periods or commas). 
    # A slight blur (Gaussian or Median) smooths out these small irregularities while 
    # keeping the larger handwriting strokes intact.
    if apply_noise_removal:
        # Using a median blur is highly effective for "salt and pepper" noise
        processed = cv2.medianBlur(processed, 3)

    # 5. Binarization / Thresholding
    # Why it matters for ICR: Many classic OCR/ICR engines (like EasyOCR) are optimized to read 
    # stark black text on a pure white background. Binarization forces every pixel to pure black or white. 
    # 
    # *Important Learning Note*: Modern Transformer-based models (like TrOCR) are often fine-tuned 
    # on natural grayscale/RGB images. Hard binarization can actually *hurt* TrOCR's accuracy 
    # because it destroys stroke thickness and anti-aliasing cues. It's a great experiment to 
    # run the pipeline with and without `--no-bin` to see how the same preprocessing helps 
    # one model but hurts another!
    if apply_binarization:
        if len(processed.shape) == 3:
             processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        # Using Otsu's thresholding which automatically finds the optimal threshold value
        # Here we use Adaptive Gaussian Thresholding for better local handling
        processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)

    # 6. Deskewing (Rotation correction)
    # Why it matters for ICR: Cursive and print recognition models assume text is aligned horizontally. 
    # Slanted lines can cause bounding box overlaps and character misalignments. 
    # Deskewing detects the text angle and rotates the image to make text horizontal.
    if apply_deskew:
        if len(processed.shape) == 3:
             gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
             gray = processed
             
        # Invert the image (so text is white, background is black) to find coordinates
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            # cv2.minAreaRect returns values in the range [-90, 0)
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Only correct if the angle is significant
            if abs(angle) > 0.5:
                (h, w) = processed.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                # Use white background for the border
                processed = cv2.warpAffine(processed, M, (w, h), 
                                         flags=cv2.INTER_CUBIC, 
                                         borderMode=cv2.BORDER_REPLICATE)

    return img, processed

def detect_text_lines(image_np):
    """
    Custom Computer Vision line segmentation module.
    Uses OpenCV morphological operations to group text into solid horizontal lines,
    finding bounding boxes that are much cleaner than EasyOCR's default word-level boxes.
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np.copy()
        
    # Invert the image and binarize
    # Text becomes white, background becomes black
    # We apply a slight Gaussian blur to smooth noise before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Adaptive Thresholding instead of Otsu's to handle colored backgrounds (like blue post-its)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 10)
    
    # Define a highly horizontal kernel to smear the text into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    
    # Dilate to connect text horizontally
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours of the smeared lines
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out very small noise boxes
        if w > 20 and h > 10:
            # Add a bit of padding to capture ascenders/descenders
            pad_x, pad_y = 10, 10
            x_min = max(0, x - pad_x)
            y_min = max(0, y - pad_y)
            x_max = min(image_np.shape[1], x + w + pad_x)
            y_max = min(image_np.shape[0], y + h + pad_y)
            
            # Format as [tl, tr, br, bl] to match EasyOCR format used in recognize.py
            tl = [x_min, y_min]
            tr = [x_max, y_min]
            br = [x_max, y_max]
            bl = [x_min, y_max]
            bboxes.append([tl, tr, br, bl])
            
    # Sort bounding boxes top-to-bottom based on y-coordinate
    bboxes = sorted(bboxes, key=lambda b: b[0][1])
    return bboxes
