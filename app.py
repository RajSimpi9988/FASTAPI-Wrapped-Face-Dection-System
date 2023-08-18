from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from PIL import Image
import io
import webbrowser  # Import the webbrowser module

app = FastAPI()

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define endpoint for face detection
@app.post("/detect_face/")
async def detect_face(file: UploadFile):
    # Read the uploaded image
    image = Image.open(io.BytesIO(await file.read()))

    # Convert PIL image to OpenCV format (BGR)
    cv_image = np.array(image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    # Convert to grayscale for face detection
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Perform face detection using Haarcascade
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Extract bounding box coordinates
    boxes = [[int(x), int(y), int(x + w), int(y + h)] for (x, y, w, h) in faces]

    # Return bounding box coordinates
    return {"boxes": boxes}
# Open the FastAPI documentation page in the default web browser
def open_browser():
    webbrowser.open("http://localhost:8000")

# Run the server and open the browser
if __name__ == "__main__":
    open_browser()  # Open the browser
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Start the server