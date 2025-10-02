import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np

# Set path to Tesseract executable (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def detect_plate_number(image_path, debug=False):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    plate_contour = None
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:  # Likely a rectangle
            plate_contour = approx
            break

    if plate_contour is not None:
        # Draw bounding box on original image
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate_image = gray[y:y + h, x:x + w]

        # Preprocess for OCR
        plate_image = cv2.resize(plate_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # enlarge
        plate_image = cv2.bilateralFilter(plate_image, 11, 17, 17)  # denoise
        _, thresh = cv2.threshold(plate_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # OCR with better config
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        plate_number = pytesseract.image_to_string(thresh, config=custom_config).strip()

        if debug:
            # Show results
            debug_img = image.copy()
            cv2.drawContours(debug_img, [plate_contour], -1, (0, 255, 0), 2)
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Detected Plate Number")
            plt.imshow(thresh, cmap='gray')
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("Bounding Box")
            plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

        return plate_number
    else:
        return "Plate not detected"


# Run detections
image_path = "car3.jpg"
plate_number = detect_plate_number(image_path, debug=True)
print("Detected Plate Number:", plate_number)