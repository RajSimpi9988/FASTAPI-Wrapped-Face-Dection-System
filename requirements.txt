Step-by-Step Instructions
Create a New Directory for the Project

Open your command prompt or terminal and navigate to the directory where you want to create your project folder: 

cd path/to/your/folder


Create a new directory for your project, for example:

mkdir face_detection_app cd face_detection_app

Set Up a Virtual Environment
Create and activate a virtual environment to isolate your project dependencies:


python -m venv venv


Activate the virtual environment:
On Windows:

venv\Scripts\activate

Install Required Libraries
Install the necessary libraries using pip:

pip install fastapi uvicorn opencv-python-headless Pillow

Run the FastAPI Application
In your terminal, navigate to the project directory and activate the virtual environment:

cd path/to/your/folder/face_detection_app
venv\Scripts\activate  # On Windows

Run the FastAPI server:

uvicorn app:app --host 0.0.0.0 --port 8000


Access the FastAPI Documentation
Open your web browser and go to http://localhost:8000/docs. 

This URL will take you to the automatically generated FastAPI documentation page.
Test the Face Detection Endpoint

On the documentation page, find the /detect_face/ route and click on it to expand.

Use the "Choose File" button to upload an image containing faces.

Click the "Execute" button to send the request and view the response with detected face bounding box coordinates.
Stop the Server

To stop the FastAPI server, press CTRL+C in the terminal where the server is running.



