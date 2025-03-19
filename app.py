import cv2
import numpy as np
import pytesseract
import os
import difflib
from flask import Flask, render_template, request
from flask_cors import CORS
from PIL import Image

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Ensure folders exist
os.makedirs("static/uploads", exist_ok=True)

# Tablet Database (Modify with more tablets)
tablet_data = {
    "DOLO-650": "Used to reduce fever and relieve mild to moderate pain.",
    "PARACETAMOL 500MG": "Pain relief and fever reduction.",
    "CROCIN": "Helps in reducing fever and mild pain.",
    "ASPIRIN": "Used to relieve pain and inflammation.",
    "IBUPROFEN": "Used to relieve pain, swelling, and fever.",
    "COMBIFLAM": "Pain relief medication, combines ibuprofen and paracetamol.",
}

def preprocess_image(image_path):
    """ Apply preprocessing steps to improve OCR accuracy """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    
    # Apply Gaussian Blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive Thresholding for contrast enhancement
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Save preprocessed image for debugging
    processed_path = os.path.join("static/uploads", "processed.jpg")
    cv2.imwrite(processed_path, img)

    return processed_path

def extract_text(image_path):
    """ Extract text from the preprocessed image """
    processed_path = preprocess_image(image_path)
    text = pytesseract.image_to_string(Image.open(processed_path), config="--psm 6")
    return text.strip()

def clean_text(text):
    """ Cleans extracted text """
    text = text.replace("\n", " ").replace("|", "").strip()
    return " ".join(text.split()).upper()

def match_tablet_name(extracted_text):
    """ Matches extracted text with known tablet names """
    extracted_words = extracted_text.split()
    for word in extracted_words:
        best_match = difflib.get_close_matches(word, tablet_data.keys(), n=1, cutoff=0.6)
        if best_match:
            return best_match[0]
    return "Unknown"

@app.route('/')
def home():
    return render_template('index.html', tablet_name=None, tablet_uses=None, image_path=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return render_template('index.html', error="No image uploaded", tablet_name=None, tablet_uses=None, image_path=None)

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error="No selected file", tablet_name=None, tablet_uses=None, image_path=None)

    # Save Original Image
    img_path = os.path.join("static/uploads", file.filename)
    file.save(img_path)

    # Extract and process text
    extracted_text = extract_text(img_path)
    cleaned_text = clean_text(extracted_text)

    # Identify Tablet
    tablet_name = match_tablet_name(cleaned_text)
    tablet_uses = tablet_data.get(tablet_name, "Details not found. Please check manually.")

    return render_template('index.html', tablet_name=tablet_name, tablet_uses=tablet_uses, image_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
