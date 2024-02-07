from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL = tf.keras.models.load_model("../saved_models/1")

try:
    MODEL = tf.saved_model.load("../models/1")
    print(MODEL.signatures)
except Exception as e:
    print("Error loading the model:", e)

CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_health"
]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    # image = np.array(Image.open(BytesIO(data)))
    image = np.array(Image.open(BytesIO(data)).convert('RGB'))
    

    # resize image as the model only accepts 224
    # image = image.resize((224, 224))
    
    # Convert to float32 and normalize the pixel values to the range [0, 1]
    image = image.astype(np.float32) / 255.0
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    # Accessing the model signature
    signature = MODEL.signatures["serving_default"]
    predictions = signature(inputs=tf.constant(img_batch))

    print("\n\n\n")
    print("prediction result",predictions['output_0'])
    print("\n\n\n")
    
    """
    print(predictions) gives output

    {'output_0': <tf.Tensor: shape=(1, 10), dtype=float32, numpy=
    array([[6.6739580e-05, 3.1271740e-03, 1.8913829e-04, 2.7338654e-04,
        6.9231202e-05, 3.6120699e-03, 9.1114128e-01, 5.6367026e-05,
        2.5995148e-05, 8.1438743e-02]], dtype=float32)>}

    """
    predicted_class = CLASS_NAMES[np.argmax(predictions['output_0'][0])]
    confidence = np.max(predictions['output_0'][0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)