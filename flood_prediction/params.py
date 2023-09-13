import os
import numpy as np

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")
TALAGANTE_LAT = os.environ.get("TALAGANTE_LAT")
TALAGANTE_LON= os.environ.get("TALAGANTE_LON")
VERTIENTES_LAT = os.environ.get("VERTIENTES_LAT")
VERTIENTES_LON= os.environ.get("VERTIENTES_LON")
MAIPU_LAT = os.environ.get("MAIPU_LAT")
MAIPU_LON= os.environ.get("MAIPU_LON")
BELLAVISTA_LAT = os.environ.get("BELLAVISTA_LAT")
BELLAVISTA_LON= os.environ.get("BELLAVISTA_LON")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
GCR_IMAGE = os.environ.get("GCR_IMAGE")
GCR_REGION = os.environ.get("GCR_REGION")
GCR_MEMORY = os.environ.get("GCR_MEMORY")
SERVICE_URL = os.environ.get("SERVICE_URL")
FOLD_LENGHT = os.environ.get("FOLD_LENGHT")
FOLD_STRIDE = os.environ.get("FOLD_STRIDE")
TRAIN_TEST_RATIO = os.environ.get("TRAIN_TEST_RATIO")
INPUT_LENGTH = os.environ.get("INPUT_LENGTH")
TARGET = os.environ.get("TARGET")
N_TARGETS = os.environ.get("N_TARGETS")
OUTPUT_LENGTH = os.environ.get("OUTPUT_LENGTH")
HORIZON = os.environ.get("HORIZON")
SEQUENCE_STRIDE = os.environ.get("SEQUENCE_STRIDE")

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "flood_forecast", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "flood_forecast", "training_outputs")

DTYPES_RAW = {
    "date": "datetime64",
    "T(degC)": "float64",
    "rain(mm)": "float64",
    "surf_press(hPa)": "float64",
    "wind_s(km/h)": "float64",
    "wind_dir(deg)": "float64",
    "soil_moist_0_to_7cm(m3)": "float64",
    "soil_moist_7_to_28cm(m3)": "float64",
    "radiation(W/m2)": "float64",
    "river_discharge(m3/s)": "float64",
    "target": "int64"
}

DTYPES_PROCESSED = np.float32

COORDS = {
    "tal": [TALAGANTE_LAT, TALAGANTE_LON],
    "ver": [VERTIENTES_LAT, VERTIENTES_LON],
    "mai": [MAIPU_LAT, MAIPU_LON],
    "bel": [BELLAVISTA_LAT, BELLAVISTA_LON]
}
