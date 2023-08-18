from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from PIL import Image
import io
import webbrowser
import mediapipe as mp

app = FastAPI()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

@app.post("/detect_hand/")
async def detect_hand(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    cv_image = np.array(image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    
    # Initialize the hand tracking model
    with mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5) as hands:
        results = hands.process(cv_image)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(cv_image, landmarks, mp_hands.HAND_CONNECTIONS)

    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

def open_browser():
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    open_browser()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)