import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

from flood_prediction.ml_logic.preprocessor import preprocess_features_pred
from flood_prediction.ml_logic.registry import load_model
from flood_prediction.ml_logic.data_prep import api_request_pred
from flood_prediction.params import *

app = FastAPI()

app.state.model = load_model()
print(type(app.state.model))

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/forecast")
def pred() -> dict:
    """
    Make a prediction of river discharge using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    X_pred = api_request_pred(COORDS)

    X_processed = preprocess_features_pred(X_pred)
    X_processed = tf.expand_dims(X_processed, axis=0)

    y_pred = app.state.model.predict(X_processed)

    res = y_pred[0][0]

    return {f'forecast': float(res)}

@app.get("/")
def root():

    return {"message": "Welcome to the Flood Forecast API",
             "documentation": "/docs",
             "predict_url": "/forecast"}
