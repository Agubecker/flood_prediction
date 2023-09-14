import os
import time
import pickle
import glob
import zipfile

from flood_prediction.params import *
from flood_prediction.ml_logic.model import *
from tensorflow import keras
from google.cloud import storage

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")

def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":

        model_filename = model_path.split("/")[-1]
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    return None

def load_model() -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'

    Return None (but do not Raise) if no model is found

    """
    if MODEL_TARGET == "local":
        print(f"\nLoad latest model from local registry...")

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        # most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        most_recent_model_path_on_disk = os.path.join(local_model_directory, "model_14.68_-321.84") # change it here to select another model

        print(f"\nLoad latest model from disk...")

        latest_model = keras.models.load_model(most_recent_model_path_on_disk, custom_objects={'nse':nse})

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        print(f"\nLoad latest model from GCS...")

        # client = storage.Client()
        # blobs = list(client.get_bucket(BUCKET_NAME).list_blobs())
        # print("###############################################")
        # try:
        #     latest_blob = max(blobs, key=lambda x: x.updated)
        #     latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, 'models', 'zip_models',latest_blob.name)
        #     latest_blob.download_to_filename(latest_model_path_to_save)

        #     # Open the ZIP file
        #     extraction_path = os.path.join(LOCAL_REGISTRY_PATH, 'models', 'unzip_models')
        #     with zipfile.ZipFile(latest_model_path_to_save, 'r') as zip_ref:
        #         zip_ref.extractall(extraction_path)
        #         local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        #         most_recent_model_path_on_disk = os.path.join(local_model_directory, "model_14.68_-321.84")
        #         latest_model = keras.models.load_model(most_recent_model_path_on_disk, custom_objects={'nse':nse})

        #     print("✅ Latest model downloaded from cloud storage")

        #     return latest_model
        # except:
        #     print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

        try:
            path_model = "model/model_14.68_-321.84"
            latest_model = keras.models.load_model(path_model, custom_objects={'nse':nse})
            print("✅ Latest model loaded")

            return latest_model
        except:
            print(f"\n❌ No model found")


        return None
    else:
        return None
