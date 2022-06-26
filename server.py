from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow
import numpy
from PIL import Image
import cv2

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

@app.get('/index')
def info(name: str):
    return f"hello {name}"

@app.post('/api/predict')
async def predict_image(imagepath:UploadFile = File(...)):
    images = []
    try:
        image = numpy.array(Image.open(imagepath.file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)
        images.append(image)
    except Exception as e:
        print(f"Broken: {imagepath}")
    
#     images = numpy.array(images, dtype = 'float32')
#     images = images / 255.0

#     class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

#     model = tensorflow.keras.models.load_model('my_model.h5')
#     predictions = model.predict(images)     # Vector of probabilities
#     pred_labels = numpy.argmax(predictions, axis = 1) # We take the highest probability
#     result = class_names[pred_labels[0]]
#     print(result)

#     # image = load_image_into_numpy_array(await image.read())
#     # print(type(image), imagepath)
    return "welcome to post!"
    # face_image = preprocessing.image.load_img(image.file_name, target_size=(150,150))

# if __name__ == "__main__":
#     uvicorn.run(app, port=8000, host='127.0.0.1')


# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pickle
# import json


# app = FastAPI()

# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class model_input(BaseModel):
    
#     Pregnancies : int
#     Glucose : int
#     BloodPressure : int
#     SkinThickness : int
#     Insulin : int
#     BMI : float
#     DiabetesPedigreeFunction :  float
#     Age : int
    

# # loading the saved model
# diabetes_model = pickle.load(open('diabetes_model.sav','rb'))


# @app.post('/diabetes_prediction')
# def diabetes_pred(input_parameters : model_input):
    
#     input_data = input_parameters.json()
#     input_dictionary = json.loads(input_data)
    
#     preg = input_dictionary['Pregnancies']
#     glu = input_dictionary['Glucose']
#     bp = input_dictionary['BloodPressure']
#     skin = input_dictionary['SkinThickness']
#     insulin = input_dictionary['Insulin']
#     bmi = input_dictionary['BMI']
#     dpf = input_dictionary['DiabetesPedigreeFunction']
#     age = input_dictionary['Age']


#     input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
#     prediction = diabetes_model.predict([input_list])
    
#     if prediction[0] == 0:
#         return 'The person is not Diabetic'
    
#     else:
#         return 'The person is Diabetic'
