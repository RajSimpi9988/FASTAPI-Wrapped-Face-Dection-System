from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils

app = FastAPI()

# Load the pre-trained face detection model
PATH_TO_SAVED_MODEL = "path/to/your/saved_model"
detection_model = tf.saved_model.load(PATH_TO_SAVED_MODEL)
detect_fn = detection_model.signatures['serving_default']

def detect_faces(image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    return detections

@app.post("/detect_faces/")
async def detect_faces_endpoint(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_np = np.array(image)
    detections = detect_faces(image_np)
    image_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np.copy(),
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(int),
        detections['detection_scores'][0].numpy(),
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.2,
        agnostic_mode=False
    )

    return Image.fromarray(image_with_detections)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
