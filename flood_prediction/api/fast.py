import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

from flood_prediction.ml_logic.preprocessor import preprocess_features_pred
from flood_prediction.ml_logic.registry import load_model
from flood_prediction.ml_logic.data_prep import api_request_pred

app = FastAPI()


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.state.model = load_model()

assert app.state.model is not None

@app.get("/forecast")
def pred() -> dict:
    """
    Make a prediction of river discharge using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    # connect it to front end and API

    X_pred = api_request_pred()

    X_processed = preprocess_features_pred(X_pred)
    X_processed = tf.expand_dims(X_processed, axis=0)
    y_pred = app.state.model.predict(X_processed)

    if y_pred[0][0] > 200:
        return {'Status':'There is an alert of flood in your area'}

    else:
        return {'Status':'There is no danger forcasted in your área'}

@app.get("/")
def root():

    return {"message": "Welcome to the Flood Forecast API",
             "documentation": "",
             "predict_url": ""}
