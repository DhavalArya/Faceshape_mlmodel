from fastapi import FastAPI, File, Form, UploadFile, Request
import fastapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import uvicorn
import tensorflow
import numpy
from PIL import Image
import cv2
import io

IMAGE_SIZE = (150,150)

app = FastAPI()

origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# define the Input class
class Input(BaseModel):
    base64str : object

def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img

@app.get('/index')
def info(name: str):
    return f"hello {name}"

@app.post('/api/predict')
async def predict_image(imagepath: bytes = File(...)):
    image = Image.open(io.BytesIO(imagepath))
    images = []
    try:
        image = numpy.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)
        images.append(image)
    except Exception as e:
        print(f"Broken: {imagepath}")

    images = numpy.array(images, dtype = 'float32')
    images = images / 255.0

    class_names = ["Heart", "Oblong", "Oval", "Round", "Square"]
    matching_names = {
        "Oval": "Varun Dhawan, Hrithik Roshan, Emma Watson, Cameron Diaz",
        "Oblong": "Akshay Kumar, Katrina Kaif",
        "Heart": "Nick Jonas, Bradley Cooper, Alia Bhatt, Deepika Padukone",
        "Round": "Shahid Kapoor, Leonardo Dicaprio, Aishwarya Rai Bacchan",
        "Square": "Aamir Khan, Tom Cruise, Jennifer Aniston"
    }

    model = tensorflow.keras.models.load_model('my_model.h5')
    predictions = model.predict(images)    
    pred_labels = numpy.argmax(predictions, axis = 1) 
    result = class_names[pred_labels[0]] + '.\\nYour face matches with ' + matching_names["Heart"]
    return result


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')
