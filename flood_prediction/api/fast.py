import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# should I add this line?
from flood_prediction.preprocessor import preprocess_features
# from taxifare.ml_logic.registry import load_model

app = FastAPI()

# we need working on "load_model" function
#app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/forecast?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/forecast")
def predict(
        date: str,
        temperature: float,
        rain: float,
        surface_pressure: float,
        radiation: float,
        windspeed: float,
        winddirection: float,
        soil_moisture_0_1cm: float,
        soil_moisture_1_3cm: float,
        soil_moisture_3_9cm: float,
        soil_moisture_9_27cm: float
    ):
    """
    Make a forecast (we decide days before and after current date).
    """
    # making prediction
    X_pred = pd.DataFrame(dict(
        date= date,
        temperature= temperature,
        rain= rain,
        surface_pressure= surface_pressure,
        radiation= radiation,
        windspeed= windspeed,
        winddirection= winddirection,
        soil_moisture_0_1cm= soil_moisture_0_1cm,
        soil_moisture_1_3cm= soil_moisture_1_3cm,
        soil_moisture_3_9cm= soil_moisture_3_9cm,
        soil_moisture_9_27cm= soil_moisture_9_27cm
    ))

    X_pred = preprocess_features(X_pred)

    # not done yet
    #y_pred = app.state.model.predict(X_pred)

    return {"fare_amount": round(float(y_pred[0][0]), 2)}


@app.get("/")
def root():
    # return {"message": "Welcome to the TaxiFareModel API",
    #         "documentation": "http://127.0.0.1:8000/docs",
    #         "predict_url": "http://127.0.0.1:8000/predict"}
    return {'greeting': 'Hello'}
