from typing import Union

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os

# Uncomment if necessary to disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = tf.keras.models.load_model("proper.h5")


@app.get("/")
def read_root():
    return {"message": "your app is working fine!!"}


@app.post("/upload")
async def create_upload_file(file: UploadFile):
    if file.size < 0:
        return {"message ": "no file uploaded"}
    try:
        contents = await file.read()

        with open(f"images/{file.filename}", "wb") as f:
            f.write(contents)

        img = cv2.imread(f"images/{file.filename}")[:, :, 0]
        img = cv2.resize(img, dsize=(28, 28), interpolation = cv2.INTER_AREA)
        img = tf.keras.utils.normalize(img, axis=1)
        # img = np.invert(np.array([img]))
        img = np.array([img]).reshape(-1, 28, 28, 1)

        prediction = model.predict(img)
        # print(prediction)
        test = np.argmax(prediction)
        return {"prediction": f"{test}"}
    except:
        return {"prediction": "Error... "}
