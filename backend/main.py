from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import math
import os

from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

# Create FastAPI app
app = FastAPI()

# Allow frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get model paths correctly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "keras_model.h5")
labels_path = os.path.join(BASE_DIR, "model", "labels.txt")

# Load detector and classifier
detector = HandDetector(maxHands=1)

classifier = Classifier(
    model_path,
    labels_path
)

# Labels list (same as your test.py)
labels = [
    "Hello",
    "I love you",
    "No",
    "OK",
    "Please",
    "Sorry",
    "Thank You",
    "Yes"
]

# Home route
@app.get("/")
def home():
    return {
        "message": "Sign Language Detection API is running"
    }


# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    offset = 20
    imgSize = 300

    hands, img = detector.findHands(img)

    if hands:

        hand = hands[0]

        x, y, w, h = hand['bbox']

        # White background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Safe crop (prevent crash)
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)

        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        aspectRatio = h / w

        if aspectRatio > 1:

            k = imgSize / h

            wCal = math.ceil(k * w)

            imgResize = cv2.resize(imgCrop, (wCal, imgSize))

            wGap = math.ceil((imgSize - wCal) / 2)

            imgWhite[:, wGap:wGap + wCal] = imgResize

        else:

            k = imgSize / w

            hCal = math.ceil(k * h)

            imgResize = cv2.resize(imgCrop, (imgSize, hCal))

            hGap = math.ceil((imgSize - hCal) / 2)

            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Prediction using SAME imgWhite as test.py
        prediction, index = classifier.getPrediction(
            imgWhite,
            draw=False
        )

        confidence = float(prediction[index])

        return {
            "prediction": labels[index],
            "confidence": confidence
        }

    return {
        "prediction": "No hand detected",
        "confidence": 0.0
    }
