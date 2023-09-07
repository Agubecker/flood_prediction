import os

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")
TALAGANTE_LAT = os.environ.get("TALAGANTE_LAT")
TALAGANTE_LON= os.environ.get("TALAGANTE_LON")
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
